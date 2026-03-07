import asyncio
import os
import uuid
import json
import yaml
import time
import csv
import argparse
import logging
import glob as glob_module
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreOutput, EngineCoreEvent

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'crawler'))
from stream_client import (
    create_stream_client,
    make_stream_request,
    update_stream_request,
)


class ANNSReplayDriver:
    """
    Driver for replaying ANNS queries where document sets (PipelinePool) change over time.
    Unlike the crawler which appends to streams, this driver updates the request with
    fresh sets of documents as the PipelinePool changes.
    """

    def __init__(self, config: dict):
        """
        config: dictionary containing model and replay configuration
        """
        self.logger = logging.getLogger(__name__)

        # Load ANNS data paths
        self.data_dir = config.get("data_dir", "data/anns")
        self.trace_dir = config.get("trace_dir",
                                    os.path.join(self.data_dir, "res"))
        self.query_trace_map_path = config.get(
            "query_trace_map",
            os.path.join(self.data_dir, "query_trace_map_5k.json"))

        # Load query trace map
        with open(self.query_trace_map_path, 'r') as f:
            self.query_trace_map = json.load(f)

        # Load document corpus from partitioned files
        self.logger.info("Loading document corpus...")
        self.document_corpus = self._load_document_corpus()
        self.logger.info(f"Loaded {len(self.document_corpus)} documents")

        # Get list of queries to replay
        # Validate mutually exclusive parameters
        if config.get("query_ids") and config.get("query_range"):
            raise ValueError(
                "query_ids and query_range are mutually exclusive. "
                "Please specify only one of them.")

        if config.get("query_ids"):
            self.query_ids = config["query_ids"]
        elif config.get("query_range"):
            # Generate query IDs from 0 to query_range-1
            query_range = config["query_range"]
            self.query_ids = list(range(query_range))
        else:
            # Use all queries from the map
            self.query_ids = sorted(
                [int(qid) for qid in self.query_trace_map.keys()])

        self.logger.info(f"Will replay {len(self.query_ids)} queries")

        self.all_input_output_details = []

        # Extract config
        self.model_config = config["model"]
        self.replay_config = config["replay"]
        self.prefix_caching_flag = config.get("prefix_caching_flag", False)
        self.log_stats = config.get("log_stats", True)

        # Logging configuration - control console verbosity
        self.logging_config = config.get("logging", {})
        self.enable_verbose_console = self.logging_config.get(
            "verbose_console", True)
        self.suppress_vllm_logs = self.logging_config.get(
            "suppress_vllm_logs", True)

        self.model_name = self.model_config["name"]
        self.seed = self.model_config.get("seed", 42)
        self.poisson_avg_arrival_time = self.replay_config[
            "poisson_avg_arrival_time"]
        self.delay_multiplier = self.replay_config.get(
            "delay_multiplier",
            1.0)  # default to 1.0 to use trace timings as-is

        self.logger.info(f"Using delay_multiplier: {self.delay_multiplier}x")

        # Setup log directory
        os.makedirs(config["config_for_run"], exist_ok=True)
        date_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        save_config_run_dir = os.path.join(config["config_for_run"], date_str)
        os.makedirs(save_config_run_dir, exist_ok=True)

        # Set up CSV logging for analysis
        self.csv_log_path = os.path.join(save_config_run_dir,
                                         'run_metrics.csv')
        self.csv_fields = [
            'event_timestamp', 'event_type', 'query_id', 'request_id',
            'stream', 'concurrency', 'duration_secs', 'details',
            'request_size', 'concurrent_requests', 'replay_rate',
            'prev_event_type', 'num_documents', 'pipeline_pool_change'
        ]
        with open(self.csv_log_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_fields)
            writer.writeheader()

        # Track request state
        self.ongoing_requests = set()
        self.current_request_tokens = {}
        self.last_event_type = {}

        # Helper method to log events in CSV format
        def log_to_csv(self,
                       event_type,
                       query_id=None,
                       request_id=None,
                       stream=None,
                       concurrency=None,
                       concurrent_requests=None,
                       duration_secs=None,
                       details=None,
                       **kwargs):
            log_entry = {
                'event_timestamp': time.time(),
                'event_type': event_type,
                'query_id': query_id,
                'request_id': request_id,
                'stream': stream,
                'concurrency': concurrency,
                'concurrent_requests': concurrent_requests,
                'duration_secs': duration_secs,
                'details': details
            }
            log_entry.update(kwargs)

            with open(self.csv_log_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_fields)
                writer.writerow(log_entry)

        self.log_to_csv = log_to_csv.__get__(self)

        # Generate or load Poisson delays
        if config.get("use_saved_poisson_delays"):
            self.poisson_delays = list(
                np.loadtxt(config["use_saved_poisson_delays"], dtype=float))
        else:
            self.poisson_delays = [
                np.random.exponential(self.poisson_avg_arrival_time)
                for _ in range(len(self.query_ids))
            ]
            save_poisson_delays_path = os.path.join(
                save_config_run_dir,
                f"poisson_avg_arrival_time_delays_{date_str}.txt")
            np.savetxt(save_poisson_delays_path, self.poisson_delays)

        self.save_config_run_dir = save_config_run_dir

        # Save config
        save_config_path = os.path.join(save_config_run_dir,
                                        f"config_{date_str}.yaml")
        with open(save_config_path, "w") as f:
            yaml.dump(config, f)

        # Sampling params
        self.sampling_params = SamplingParams(
            max_tokens=self.replay_config["max_output_tokens"],
            temperature=0.0,
            top_p=1.0,
            seed=self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # vLLM Client
        self.client = create_stream_client(
            self.model_name,
            max_num_batched_tokens=self.model_config["max_token_budget"],
            gpu_memory_utilization=self.model_config["gpu_memory_utilization"],
            tensor_parallel_size=self.model_config["tensor_parallel_size"],
            seed=self.seed,
            enable_prefix_caching=self.prefix_caching_flag,
            log_stats=self.log_stats,
        )

        self.query_tasks: List[asyncio.Task] = []

        # Templates from config
        self.template = self.replay_config["template"]
        self.end_template = self.replay_config["end_template"]

        # Add file handler for logging to run.log
        fh = logging.FileHandler(os.path.join(save_config_run_dir, 'run.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

    def _load_document_corpus(self) -> Dict[str, str]:
        """Load document corpus from partitioned JSON files."""
        corpus = {}
        pattern = os.path.join(self.data_dir,
                               "retrieved_corpus_content.part.*.json")
        corpus_files = sorted(glob_module.glob(pattern))

        if not corpus_files:
            raise FileNotFoundError(
                f"No corpus files found matching pattern: {pattern}")

        for corpus_file in corpus_files:
            self.logger.info(f"Loading corpus file: {corpus_file}")
            with open(corpus_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                corpus.update(data)

        return corpus

    def log_request_event(self, event: EngineCoreEvent, request_id: str,
                          query_id: int, stream: bool):
        """Log an engine core event with its timing information."""
        log_entry = {
            'event_type': event.type,
            'event_timestamp': event.start_timestamp,
            'duration_secs': event.duration,
            'request_id': request_id,
            'request_size':
            len(self.current_request_tokens.get(request_id, [])),
            'concurrent_requests': len(self.ongoing_requests),
            'replay_rate': 1.0 / self.poisson_avg_arrival_time,
            'query_id': query_id,
            'stream': stream,
            'prev_event_type': self.last_event_type.get(request_id),
        }
        self.last_event_type[request_id] = event.type
        self.log_to_csv(**log_entry)

    def cleanup(self):
        """Clear all state before starting a new replay configuration."""
        self.ongoing_requests.clear()
        self.current_request_tokens.clear()
        self.last_event_type.clear()
        self.query_tasks = []

    async def replay(self, concurrency: bool, stream: bool):
        """
        Replay the ANNS queries with changing document sets (PipelinePool).
        The arrival of queries is modeled by a Poisson process.
        """
        self.cleanup()

        self.logger.info(
            f"Replaying {len(self.query_ids)} ANNS queries with stream={stream}, concurrency={concurrency}"
        )
        self.log_to_csv('replay_start',
                        stream=stream,
                        concurrency=concurrency,
                        details=f"Replaying {len(self.query_ids)} queries")
        start_time = time.time_ns()

        # Create progress bar (disable if output is redirected to file)
        self.pbar = tqdm(
            total=len(self.query_ids),
            desc=
            f"Replaying ANNS queries (stream={stream}, concurrency={concurrency})",
            disable=not sys.stdout.isatty(
            )  # Disable progress bar when redirected to file
        )

        for i, query_id in enumerate(self.query_ids):
            if concurrency:
                time_to_next_query = self.poisson_delays[i]
                if self.enable_verbose_console:
                    self.pbar.write(
                        f"Sleeping for {time_to_next_query} seconds before query {query_id}"
                    )
                self.log_to_csv('query_delay',
                                query_id=query_id,
                                stream=stream,
                                concurrency=concurrency,
                                duration_secs=time_to_next_query)
                await asyncio.sleep(time_to_next_query)
                task = asyncio.create_task(self._replay_query(
                    query_id, stream))
                self.query_tasks.append(task)
            else:
                await self._replay_query(query_id, stream)

        await asyncio.gather(*self.query_tasks)
        end_time = time.time_ns()
        trace_replay_time = (end_time - start_time) / 1e9
        self.log_to_csv('replay_end',
                        stream=stream,
                        concurrency=concurrency,
                        duration_secs=trace_replay_time,
                        details=f"Finished {len(self.query_ids)} queries")
        self.pbar.close()
        self.logger.info(
            f"Finished replaying {len(self.query_ids)} queries with stream={stream}, concurrency={concurrency} in {trace_replay_time:.3f} seconds"
        )

    def _parse_pipeline_pool(self, pool_str: str) -> List[int]:
        """Parse PipelinePool string like '(62952551,79119332,83808733)' into list of doc IDs."""
        # Remove parentheses and split by comma
        pool_str = pool_str.strip('()"')
        if not pool_str:
            return []
        return [int(doc_id.strip()) for doc_id in pool_str.split(',')]

    def _build_prompt_from_docs(self, query: str, doc_ids: List[int]) -> str:
        """Build prompt from template, documents, and query."""
        # Start with template
        prompt = self.template + '\n'

        # Add documents
        for doc_id in doc_ids:
            doc_id_str = str(doc_id)
            if doc_id_str in self.document_corpus:
                prompt += self.document_corpus[doc_id_str] + '\n'
            else:
                self.logger.warning(f"Document {doc_id} not found in corpus")

        # Add end template and query
        prompt += self.end_template + ' ' + query

        return prompt

    def _extract_anns_trace(
            self, query_id: int,
            stream: bool) -> Tuple[str, List[Tuple[float, List[int]]]]:
        """
        Extract query text and timeline of PipelinePool changes from ANNS trace.

        Returns:
            (query_text, updates) where updates is a list of (delay_seconds, doc_ids)
        """
        # Get query info
        query_info = self.query_trace_map[str(query_id)]
        query_text = query_info["query"]
        trace_file = query_info["trace_file"]
        trace_path = os.path.join(self.trace_dir, trace_file)

        if not os.path.exists(trace_path):
            self.logger.error(f"Trace file not found: {trace_path}")
            return query_text, []

        # Read trace CSV
        trace_df = pd.read_csv(trace_path)

        # Parse timeline
        updates = []
        for _, row in trace_df.iterrows():
            # Delay is the duration of this time window in microseconds, convert to seconds
            delay = (row['EndTime_us'] - row['StartTime_us']) / 1e6
            # Apply delay multiplier from config
            delay = delay * self.delay_multiplier
            doc_ids = self._parse_pipeline_pool(row['PipelinePool'])
            updates.append((delay, doc_ids))

        return query_text, updates

    async def _replay_query(self, query_id: int, stream: bool):
        """Replay a single ANNS query with changing document sets."""
        request_id = str(uuid.uuid4())
        output_tokens: List[EngineCoreOutput] = []

        if self.enable_verbose_console:
            self.pbar.write(
                f"Starting ANNS query {query_id} (Request ID: {request_id})")
        self.log_to_csv('query_start',
                        query_id=query_id,
                        request_id=request_id,
                        stream=stream)

        # Extract query text and timeline of document set changes
        query_text, updates = self._extract_anns_trace(query_id, stream)

        if not updates:
            self.logger.warning(f"No updates found for query {query_id}")
            return

        timer_start = time.time_ns()
        self.ongoing_requests.add(request_id)

        if stream:
            # Streaming mode: start with first document set, then update as PipelinePool changes

            # First update: build initial prompt
            first_delay, first_doc_ids = updates[0]
            first_prompt = self._build_prompt_from_docs(
                query_text, first_doc_ids)
            first_prompt_tokens = self.tokenizer.encode(first_prompt)[
                1:]  # Skip BOS token

            self.current_request_tokens[request_id] = first_prompt_tokens

            # Wait for first time window
            if self.enable_verbose_console:
                self.pbar.write(
                    f"STREAM={stream} Query {query_id} (Request ID: {request_id}) Sleeping for {first_delay} seconds"
                )
            self.log_to_csv('query_sleep',
                            query_id=query_id,
                            request_id=request_id,
                            stream=stream,
                            duration_secs=first_delay,
                            num_documents=len(first_doc_ids))
            await asyncio.sleep(first_delay)

            # Create initial streaming request
            stream_request = make_stream_request(
                self.sampling_params,
                request_id,
                first_prompt_tokens,
                is_streaming_prompt=True,
                is_streaming_prompt_finished=(
                    len(updates) == 1),  # Finished if only one update
            )

            input_output_pair = await self.client.create_stream_request_async(
                stream_request)
            q: asyncio.Queue[EngineCoreRequest] = input_output_pair[0]

            # Process remaining updates (if any)
            for i in range(1, len(updates)):
                delay, doc_ids = updates[i]
                is_last = (i == len(updates) - 1)

                await asyncio.sleep(delay)

                # Build new prompt with updated document set
                new_prompt = self._build_prompt_from_docs(query_text, doc_ids)
                new_prompt_tokens = self.tokenizer.encode(new_prompt)[1:]

                # Update token count
                self.current_request_tokens[request_id] = new_prompt_tokens

                # Send update request (not append!)
                update_req = update_stream_request(
                    request_id,
                    new_prompt_tokens,
                    is_streaming_prompt=True,
                    is_streaming_prompt_finished=is_last)

                self.log_to_csv('pipeline_pool_update',
                                query_id=query_id,
                                request_id=request_id,
                                stream=stream,
                                duration_secs=delay,
                                num_documents=len(doc_ids),
                                pipeline_pool_change=True)

                await q.put(update_req)

            # Signal end of stream
            await q.put(None)

            # Collect outputs
            start_ttft = time.time_ns()
            first_decode_token: EngineCoreOutput = await self.client.get_output_async(
                request_id)
            ttft = (time.time_ns() - start_ttft) / 1e9
            output_tokens.append(first_decode_token)

            accumulated_output_token_ids = first_decode_token.new_token_ids
            current_token_for_loop = first_decode_token

            while current_token_for_loop.finish_reason is None and len(
                    accumulated_output_token_ids
            ) < self.replay_config["max_output_tokens"]:
                current_token_for_loop: EngineCoreOutput = await self.client.get_output_async(
                    request_id)
                output_tokens.append(current_token_for_loop)
                accumulated_output_token_ids.extend(
                    current_token_for_loop.new_token_ids)

            # Extract num_tokens_invalidated from the final output (if available)
            num_tokens_invalidated = None
            if output_tokens and hasattr(output_tokens[-1],
                                         'num_tokens_invalidated'):
                num_tokens_invalidated = output_tokens[
                    -1].num_tokens_invalidated

            self.all_input_output_details.append({
                "query_id":
                str(query_id),
                "request_id":
                request_id,
                "prompt_token_ids":
                list(self.current_request_tokens[request_id]),
                "output_token_ids":
                accumulated_output_token_ids,
                "output_text":
                self.tokenizer.decode(accumulated_output_token_ids),
                "is_streaming":
                stream,
                "num_pipeline_updates":
                len(updates),
                "num_tokens_invalidated":
                num_tokens_invalidated
            })

        else:
            # Non-streaming baseline: wait for all updates, then send final prompt

            # Wait through all time windows
            total_delay = sum(delay for delay, _ in updates)
            if self.enable_verbose_console:
                self.pbar.write(
                    f"BASELINE={stream} Query {query_id} (Request ID: {request_id}) Sleeping for {total_delay} seconds"
                )
            self.log_to_csv('query_sleep',
                            query_id=query_id,
                            request_id=request_id,
                            stream=stream,
                            duration_secs=total_delay)
            await asyncio.sleep(total_delay)

            # Use final document set
            _, final_doc_ids = updates[-1]
            final_prompt = self._build_prompt_from_docs(
                query_text, final_doc_ids)
            final_prompt_tokens = self.tokenizer.encode(final_prompt)[1:]

            self.current_request_tokens[request_id] = final_prompt_tokens

            # Create non-streaming request
            stream_request = make_stream_request(
                self.sampling_params,
                request_id,
                final_prompt_tokens,
                is_streaming_prompt=False,
                is_streaming_prompt_finished=False,
            )

            input_output_pair = await self.client.create_stream_request_async(
                stream_request)
            q: asyncio.Queue[EngineCoreRequest] = input_output_pair[0]
            await q.put(None)

            # Collect outputs
            start_ttft = time.time_ns()
            first_decode_token: EngineCoreOutput = await self.client.get_output_async(
                request_id)
            ttft = (time.time_ns() - start_ttft) / 1e9
            output_tokens.append(first_decode_token)

            accumulated_output_token_ids = first_decode_token.new_token_ids
            current_token_for_loop = first_decode_token

            while current_token_for_loop.finish_reason is None and len(
                    accumulated_output_token_ids
            ) < self.replay_config["max_output_tokens"]:
                current_token_for_loop: EngineCoreOutput = await self.client.get_output_async(
                    request_id)
                output_tokens.append(current_token_for_loop)
                accumulated_output_token_ids.extend(
                    current_token_for_loop.new_token_ids)

            # Extract num_tokens_invalidated from the final output (if available)
            num_tokens_invalidated = None
            if output_tokens and hasattr(output_tokens[-1],
                                         'num_tokens_invalidated'):
                num_tokens_invalidated = output_tokens[
                    -1].num_tokens_invalidated

            self.all_input_output_details.append({
                "query_id":
                str(query_id),
                "request_id":
                request_id,
                "prompt_token_ids":
                list(final_prompt_tokens),
                "output_token_ids":
                accumulated_output_token_ids,
                "output_text":
                self.tokenizer.decode(accumulated_output_token_ids),
                "is_streaming":
                stream,
                "num_pipeline_updates":
                len(updates),
                "num_tokens_invalidated":
                num_tokens_invalidated
            })

        # Log TTFT and E2E latency
        self.log_to_csv('query_ttft',
                        query_id=query_id,
                        request_id=request_id,
                        stream=stream,
                        duration_secs=ttft)
        if self.enable_verbose_console:
            self.pbar.write(
                f"Query TTFT {query_id} (Request ID: {request_id}) completed in {ttft:.3f} seconds"
            )

        e2e_query_time = (time.time_ns() - timer_start) / 1e9
        if self.enable_verbose_console:
            self.pbar.write(
                f"Query E2E {query_id} (Request ID: {request_id}) completed in {e2e_query_time:.3f} seconds"
            )
        self.log_to_csv('query_e2e_latency',
                        query_id=query_id,
                        request_id=request_id,
                        stream=stream,
                        duration_secs=e2e_query_time)

        # Log all engine core events
        for output in output_tokens:
            if output.events is not None:
                for event in output.events:
                    self.log_request_event(event, request_id, query_id, stream)

        # Clean up request state
        self.ongoing_requests.discard(request_id)
        self.current_request_tokens.pop(request_id, None)

        # Update progress bar
        self.pbar.update(1)
        self.pbar.refresh()


async def test_main(args: argparse.Namespace):
    config_path = args.config

    # Load configuration (YAML)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set environment variables based on config
    os.environ["USE_SWAP_LATENCY_PREDICTOR"] = str(
        config.get("use_swap_latency_predictor", 0))
    os.environ["SAVE_SWAP_LATENCY_DATA_PATH"] = config.get(
        "save_swap_latency_data_path", "")
    os.environ["USE_SAVED_SWAP_LATENCY_DATA_PATH"] = config.get(
        "use_saved_swap_latency_data_path", "")

    os.environ["USE_RECOMPUTATION_LATENCY_PREDICTOR"] = str(
        config.get("use_recomputation_latency_predictor", 0))
    os.environ["SAVE_RECOMPUTATION_DATA_PATH"] = config.get(
        "save_recomputation_data_path", "")
    os.environ["USE_SAVED_RECOMPUTATION_DATA_PATH"] = config.get(
        "use_saved_recomputation_data_path", "")

    os.environ["SCHEDULER_TYPE"] = str(config.get("scheduler", "default_vllm"))

    # Suppress vLLM logging via environment variable (affects child processes too)
    suppress_vllm = config.get("logging", {}).get("suppress_vllm_logs", True)
    if suppress_vllm:
        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
        # Also suppress transformers and tokenizers verbosity
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ[
            "TOKENIZERS_PARALLELISM"] = "false"  # Reduces tokenizer warnings

    # Validate environment variables
    if os.environ["USE_SAVED_SWAP_LATENCY_DATA_PATH"] and os.environ[
            "SAVE_SWAP_LATENCY_DATA_PATH"]:
        raise ValueError(
            "Cannot use both USE_SAVED_SWAP_LATENCY_DATA_PATH and SAVE_SWAP_LATENCY_DATA_PATH"
        )
    if os.environ["USE_SAVED_RECOMPUTATION_DATA_PATH"] and os.environ[
            "SAVE_RECOMPUTATION_DATA_PATH"]:
        raise ValueError(
            "Cannot use both USE_SAVED_RECOMPUTATION_DATA_PATH and SAVE_RECOMPUTATION_DATA_PATH"
        )

    driver = ANNSReplayDriver(config=config)

    # Suppress vLLM logging if configured (after driver creation to catch all loggers)
    if driver.suppress_vllm_logs:
        # Set vLLM and related libraries to WARNING level to reduce noise
        # Note: This affects the main process only; child processes (tensor parallel workers)
        # may still emit logs unless vLLM's logging is patched in the source
        for logger_name in ["vllm", "transformers", "torch", "ray"]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            # Also set all existing handlers to WARNING
            for handler in logger.handlers:
                handler.setLevel(logging.WARNING)

        # Suppress all child loggers of vllm recursively
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name.startswith(("vllm", "ray", "torch")):
                logging.getLogger(name).setLevel(logging.WARNING)

    # Run experiments from config
    for experiment in config["experiments"]:
        await driver.replay(concurrency=experiment["concurrency"],
                            stream=experiment["stream"])

    # Save all collected prompt and output details
    if driver.all_input_output_details:
        streaming_details = []
        non_streaming_details = []
        for item in driver.all_input_output_details:
            if item.get('is_streaming', False):
                streaming_details.append(item)
            else:
                non_streaming_details.append(item)

        if streaming_details:
            output_data_path_streaming = os.path.join(
                driver.save_config_run_dir, "collected_outputs_streaming.json")
            try:
                with open(output_data_path_streaming, "w") as f:
                    json.dump(streaming_details, f, indent=2)
                driver.logger.info(
                    f"Saved {len(streaming_details)} streaming details to {output_data_path_streaming}"
                )
            except Exception as e:
                driver.logger.error(
                    f"Failed to save collected_outputs_streaming.json: {e}")

        if non_streaming_details:
            output_data_path_non_streaming = os.path.join(
                driver.save_config_run_dir,
                "collected_outputs_non_streaming.json")
            try:
                with open(output_data_path_non_streaming, "w") as f:
                    json.dump(non_streaming_details, f, indent=2)
                driver.logger.info(
                    f"Saved {len(non_streaming_details)} non-streaming details to {output_data_path_non_streaming}"
                )
            except Exception as e:
                driver.logger.error(
                    f"Failed to save collected_outputs_non_streaming.json: {e}"
                )
    else:
        driver.logger.info("No prompt details collected to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        help="Path to configuration YAML file",
                        default="experiments/anns/quick_config_replay_anns.yaml")
    args = parser.parse_args()

    # Load config early to check logging settings
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Configure logging based on config
    suppress_vllm = config.get("logging", {}).get("suppress_vllm_logs", True)

    if suppress_vllm:
        # Set up logging with specific levels for different loggers
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True  # Override any existing configuration
        )
        # Suppress noisy loggers BEFORE any vLLM imports
        logging.getLogger("vllm").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("ray").setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    asyncio.run(test_main(args))
