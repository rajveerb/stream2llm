import asyncio
from glob import glob
from typing import Dict, List, Tuple
import uuid
import json
import yaml
import time  # Added for debugging/logging if needed
import os
import argparse
import logging
import csv
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm

from transformers import AutoTokenizer  # Added import

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreOutput, EngineCoreEvent
from stream_client import (
    create_stream_client,
    make_stream_request,
    append_stream_request,
)


class QueryReplayDriver:
    """
    Driver for replaying retrieval crawls for a list of queries using vLLM streaming.
    Arrival of crawl requests are modeled as a Poisson process.
    """

    def __init__(
        self,
        config: dict,
    ):
        """
        config: dictionary containing model and replay configuration
        """
        self.logger = logging.getLogger(__name__)
        trace_dir = config["trace_dir"]
        if not os.path.exists(trace_dir):
            self.logger.error(f"Error: Trace directory not found: {trace_dir}")
            return
        self.traces_dir = trace_dir
        self.traces = glob(f"{trace_dir}/query_*.csv")
        # sort traces by query id
        self.traces.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.logger.info(f"Sorted {self.traces}")
        self.logger.info(f"Found {len(self.traces)} traces")

        self.all_input_output_details = []

        # Extract config
        self.model_config = config["model"]
        self.replay_config = config["replay"]
        self.prefix_caching_flag = config["prefix_caching_flag"]
        self.log_stats = config["log_stats"]

        self.model_name = self.model_config["name"]
        self.seed = self.model_config.get("seed", 42)  # Add seed extraction
        self.poisson_avg_arrival_time = self.replay_config[
            "poisson_avg_arrival_time"]

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
            'prev_event_type'
        ]
        with open(self.csv_log_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_fields)
            writer.writeheader()

        # Track request state
        self.ongoing_requests = set()  # Track active requests
        self.current_request_tokens = {}  # request_id -> token count
        self.last_event_type = {}  # request_id -> last event type

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

            # Include additional fields from kwargs
            log_entry.update(kwargs)

            with open(self.csv_log_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_fields)
                writer.writerow(log_entry)

        # Add the method to the class
        self.log_to_csv = log_to_csv.__get__(self)

        if config.get("use_saved_poisson_delays"):
            self.poisson_delays = list(
                np.loadtxt(config["use_saved_poisson_delays"], dtype=float))
        else:
            self.poisson_delays = [
                np.random.exponential(self.poisson_avg_arrival_time)
                for _ in range(len(self.traces))
            ]
            save_poisson_delays_path = os.path.join(
                save_config_run_dir,
                f"poisson_avg_arrival_time_delays_{date_str}.txt")
            np.savetxt(save_poisson_delays_path, self.poisson_delays)

        self.save_config_run_dir = save_config_run_dir  # Store for later use

        # save config
        save_config_path = os.path.join(save_config_run_dir,
                                        f"config_{date_str}.yaml")
        with open(save_config_path, "w") as f:
            yaml.dump(config, f)

        self.sampling_params = SamplingParams(
            max_tokens=self.replay_config["max_output_tokens"],
            temperature=0.0,  # Set temperature to 0 for deterministic sampling
            top_p=1.0,  # Use all tokens in the distribution
            seed=self.seed  # Set a fixed seed for reproducibility
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # vLLM Client and state
        self.client: AsyncMPClient = create_stream_client(
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
        self.query_id_to_prompt: Dict[int, str] = {}

        # Add file handler for logging to run.log
        fh = logging.FileHandler(os.path.join(save_config_run_dir, 'run.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

    def log_request_event(self, event: EngineCoreEvent, request_id: str,
                          query_id: int, stream: bool):
        """Log an engine core event with its timing information."""
        log_entry = {
            'event_type': event.type,
            'event_timestamp':
            event.start_timestamp,  # Use start_timestamp instead of timestamp
            'duration_secs':
            event.duration,  # Use the duration directly from event
            'request_id': request_id,
            'request_size':
            len(self.current_request_tokens.get(request_id, [])),
            'concurrent_requests': len(self.ongoing_requests),
            'replay_rate': 1.0 / self.poisson_avg_arrival_time,
            'query_id': query_id,
            'stream': stream,
            'prev_event_type': self.last_event_type.get(request_id),
        }
        # Update tracking state
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
        Replay the queries' retrieval traces. The arrival of queries is modeled
        by a Poisson process.
        """
        # Clear state before starting new replay
        self.cleanup()

        self.logger.info(
            f"Replaying {len(self.traces)} traces with stream={stream}, concurrency={concurrency}"
        )
        self.log_to_csv('replay_start',
                        stream=stream,
                        concurrency=concurrency,
                        details=f"Replaying {len(self.traces)} traces")
        start_time = time.time_ns()

        # Create progress bar and store as instance variable
        self.pbar = tqdm(
            total=len(self.traces),
            desc=
            f"Replaying queries (stream={stream}, concurrency={concurrency})")

        for i, trace_fp in enumerate(self.traces):
            if concurrency:
                time_to_next_query = self.poisson_delays[i]
                self.pbar.write(
                    f"Sleeping for {time_to_next_query} seconds before query {i}"
                )
                self.log_to_csv('query_delay',
                                query_id=i,
                                stream=stream,
                                concurrency=concurrency,
                                duration_secs=time_to_next_query)
                await asyncio.sleep(time_to_next_query)
                task = asyncio.create_task(
                    self._replay_trace(trace_fp, i, stream))
                self.query_tasks.append(task)
            else:
                await self._replay_trace(trace_fp, i, stream)

        await asyncio.gather(*self.query_tasks)
        end_time = time.time_ns()
        trace_replay_time = (end_time - start_time) / 1e9
        self.pbar.write(
            f"Finished replaying {len(self.traces)} traces with stream={stream}, concurrency={concurrency} in {trace_replay_time:.3f} seconds"
        )
        self.log_to_csv('replay_end',
                        stream=stream,
                        concurrency=concurrency,
                        duration_secs=trace_replay_time,
                        details=f"Finished {len(self.traces)} traces")
        self.pbar.close()

    def extract_delays(self, trace_fp: str,
                       stream: bool) -> Tuple[List[Tuple[int, int]], int, str]:
        # Read the entire trace file
        trace_df = pd.read_csv(trace_fp)
        # if key content not found print the trace_fp
        if 'content' not in trace_df.columns:
            # Example of a case where this might happen:
            # When links returned by tavily are PDFs or domains that are not whitelisted
            self.logger.error(f"Content not found in trace file: {trace_fp}")
            self.log_to_csv(
                'error',
                details=f"Content not found in trace file: {trace_fp}")
            return [], 0, ""  # Return empty delays, zero sleep time, and empty query

        trace_df['content'] = trace_df['content'].fillna('')
        # Sort by endTime to ensure delays are calculated in the correct order
        trace_df = trace_df.sort_values('endTime').reset_index(drop=True)

        # Extract query from the first tavily_search row
        # There should be only one query per trace
        tavily_row = trace_df[trace_df['type'] == 'tavily_search']
        query = tavily_row['query'].iloc[0] if not tavily_row.empty else None

        # Encode content for page_scrape rows
        # Only page_scrape rows have content to encode
        trace_df['encoded_content'] = trace_df.apply(
            lambda row: self.tokenizer.encode(row['content'] + '\n')[1:]
            if row['type'] == 'page_scrape' else None,
            axis=1)

        # Compute delays between consecutive events
        # For each page_scrape row (except the first), compute delay as the difference between its endTime and the previous event's endTime
        delays = []
        prev_end_time = None
        for _, row in trace_df.iterrows():
            if prev_end_time is not None and row['type'] == 'page_scrape':
                delay = row['endTime'] - prev_end_time
                delays.append((delay, row['encoded_content']))
            prev_end_time = row['endTime']

        # Determine first_sleep value based on streaming mode
        # if stream then sleep for tavily search else sleep for until web crawl finishes for the given query
        first_sleep = trace_df['endTime'].iloc[0] if stream else trace_df[
            'endTime'].iloc[-1]

        return delays, first_sleep, query

    async def _replay_trace(self, trace_fp: str, query_id: int, stream: bool):
        """Initiate a stream request and TraceReplayClient for the query"""
        # Generate unique VLLM request ID
        request_id = str(uuid.uuid4())
        output_tokens: EngineCoreOutput = []

        self.pbar.write(
            f"Starting query {query_id} (Request ID: {request_id}) from {trace_fp}"
        )
        self.log_to_csv('query_start',
                        query_id=query_id,
                        request_id=request_id,
                        stream=stream,
                        details=f"From {trace_fp}")

        delays, first_sleep, query = self.extract_delays(trace_fp, stream)

        if delays == []:
            return

        tokenized_template_before_context = self.tokenizer.encode(
            self.template + '\n')
        # Track token count for this request
        self.current_request_tokens[
            request_id] = tokenized_template_before_context

        timer_start = time.time_ns()
        self.pbar.write(
            f"STREAM={stream} Query {query_id} (Request ID: {request_id}) Sleeping for {first_sleep} seconds"
        )
        self.log_to_csv('query_sleep',
                        query_id=query_id,
                        request_id=request_id,
                        stream=stream,
                        duration_secs=first_sleep)
        await asyncio.sleep(
            first_sleep
        )  # captures tavily search in secs if stream else sleep for entire crawl's duration in secs

        self.ongoing_requests.add(request_id)
        if stream:
            stream_request = make_stream_request(
                self.sampling_params,
                request_id,
                tokenized_template_before_context,
                is_streaming_prompt=True,
                is_streaming_prompt_finished=False,
            )

            input_output_pair = await self.client.create_stream_request_async(
                stream_request)
            q: asyncio.Queue[EngineCoreRequest] = input_output_pair[0]

            for i in range(len(delays) - 1):
                delay, content = delays[i]
                await asyncio.sleep(delay)
                # Update token count with new content
                self.current_request_tokens[request_id].extend(content)
                stream_request = append_stream_request(
                    request_id,
                    content,
                    is_streaming_prompt=True,
                    is_streaming_prompt_finished=False)
                await q.put(stream_request)

            # Send the final content
            delay, content = delays[-1]
            await asyncio.sleep(delay)
            final_content = content + self.tokenizer.encode(self.end_template +
                                                            ' ' + query)[1:]
            # Update token count with final content
            self.current_request_tokens[request_id].extend(final_content)
            stream_request = append_stream_request(
                request_id,
                final_content,
                is_streaming_prompt=True,
                is_streaming_prompt_finished=True)
            await q.put(stream_request)
            # Signal end of this stream
            await q.put(None)
            start_ttft = time.time_ns()
            first_decode_token: EngineCoreOutput = await self.client.get_output_async(
                request_id)
            ttft = (time.time_ns() - start_ttft) / 1e9
            output_tokens.append(first_decode_token)

            # --- Collect prompt and output details for streaming ---
            current_prompt_ids = list(
                self.current_request_tokens[request_id])  # Use tracked tokens

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

            self.all_input_output_details.append({
                "query_id":
                str(query_id),
                "request_id":
                request_id,
                "prompt_token_ids":
                current_prompt_ids,  # This was defined before this replaced block
                "output_token_ids":
                accumulated_output_token_ids,
                "output_text":
                self.tokenizer.decode(accumulated_output_token_ids),
                "is_streaming":
                stream  # Use the 'stream' variable passed to _replay_trace
            })
        else:

            # entire_prompt will be sent in one go
            prompt_ids = tokenized_template_before_context

            for i in range(len(delays)):
                _, content = delays[i]
                prompt_ids.extend(content)
            prompt_ids.extend(
                self.tokenizer.encode(self.end_template + ' ' + query)[1:])

            # Update token count with all content
            self.current_request_tokens[request_id] = prompt_ids

            stream_request = make_stream_request(
                self.sampling_params,
                request_id,
                prompt_ids,
                is_streaming_prompt=False,
                is_streaming_prompt_finished=False,
            )

            input_output_pair = await self.client.create_stream_request_async(
                stream_request)
            q: asyncio.Queue[EngineCoreRequest] = input_output_pair[0]

            await q.put(None)

            start_ttft = time.time_ns()
            first_decode_token: EngineCoreOutput = await self.client.get_output_async(
                request_id)
            ttft = (time.time_ns() - start_ttft) / 1e9
            output_tokens.append(first_decode_token)

            # --- Collect prompt and output details for non-streaming ---
            # prompt_ids is already the full prompt here
            current_prompt_ids = list(prompt_ids)

            accumulated_output_token_ids = first_decode_token.new_token_ids
            current_token_for_loop = first_decode_token  # 'first_decode_token' holds the first_chunk from the get_output_async call right before this block

            while current_token_for_loop.finish_reason is None and len(
                    accumulated_output_token_ids
            ) < self.replay_config["max_output_tokens"]:
                current_token_for_loop: EngineCoreOutput = await self.client.get_output_async(
                    request_id)
                output_tokens.append(current_token_for_loop)
                accumulated_output_token_ids.extend(
                    current_token_for_loop.new_token_ids)

            self.all_input_output_details.append({
                "query_id":
                str(query_id),
                "request_id":
                request_id,
                "prompt_token_ids":
                current_prompt_ids,  # This was defined before this replaced block
                "output_token_ids":
                accumulated_output_token_ids,
                "output_text":
                self.tokenizer.decode(accumulated_output_token_ids),
                "is_streaming":
                stream  # Use the 'stream' variable passed to _replay_trace
            })

        self.log_to_csv('query_ttft',
                        query_id=query_id,
                        request_id=request_id,
                        stream=stream,
                        duration_secs=ttft)
        self.pbar.write(
            f"Query TTFT {query_id} (Request ID: {request_id}) completed in {ttft:.3f} seconds"
        )
        e2e_query_time = (time.time_ns() - timer_start) / 1e9
        self.pbar.write(
            f"Query E2E {query_id} (Request ID: {request_id}) completed in {e2e_query_time:.3f} seconds"
        )
        self.log_to_csv('query_e2e_latency',
                        query_id=query_id,
                        request_id=request_id,
                        stream=stream,
                        duration_secs=e2e_query_time)

        for output in output_tokens:
            if output.events is not None:  # Add check for None
                for event in output.events:
                    self.log_request_event(event, request_id, query_id, stream)

        # Clean up request state
        self.ongoing_requests.discard(request_id)
        self.current_request_tokens.pop(request_id, None)

        # Update progress bar with metadata
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

    # TODO: rbachkaniwala3: Currently, the only scheduler supported is default_vllm.
    os.environ["SCHEDULER_TYPE"] = str(config.get("scheduler", "default_vllm"))

    # can't use saved and save at the same time
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

    driver = QueryReplayDriver(config=config, )

    # Run experiments from config
    for experiment in config["experiments"]:
        await driver.replay(concurrency=experiment["concurrency"],
                            stream=experiment["stream"])

    # Save all collected prompt and output details
    if driver.all_input_output_details:
        streaming_details = []
        non_streaming_details = []
        for item in driver.all_input_output_details:
            # Ensure the 'is_streaming' key exists, default to False or log if critical
            if item.get('is_streaming', False):
                streaming_details.append(item)
            else:
                non_streaming_details.append(item)

        if streaming_details:
            output_data_path_streaming = os.path.join(
                driver.save_config_run_dir, "collected_outputs_streaming.json")
            try:
                with open(output_data_path_streaming, "w") as f:
                    json.dump(
                        streaming_details,
                        f,
                    )
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
                    json.dump(
                        non_streaming_details,
                        f,
                    )
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
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
        default="experiments/crawler/quick_config_replay_web_crawl.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(test_main(args))
