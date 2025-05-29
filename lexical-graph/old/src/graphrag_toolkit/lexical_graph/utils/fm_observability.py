# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import time
import logging
import threading
import queue
from multiprocessing import Queue
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Callable, cast

from llama_index.core import Settings
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload, CBEvent
from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.utilities.token_counting import TokenCounter
from llama_index.core.callbacks.token_counting import TokenCountingEvent

logger = logging.getLogger(__name__)

_fm_observability_queue = None


class FMObservabilityQueuePoller(threading.Thread):
    """
    Manages a thread-based queue poller for processing observability events.

    This class runs in a separate thread to process events from a queue. It interacts
    with an observability statistics tracker and can be safely started and stopped
    to manage the lifecycle of event processing.

    :ivar _discontinue: A threading event used to signal and manage the stop state
        of the thread. It enables controlled termination of the thread loop.
    :type _discontinue: threading.Event
    :ivar fm_observability: Tracks and manages observability-related statistics,
        providing functionality to handle processed events.
    :type fm_observability: FMObservabilityStats
    """

    def __init__(self):
        """
        Represents the initialization of an instance for handling internal
        state and observability statistics.

        :ivar _discontinue: A threading event object used to signal when
            the operation should be discontinued.
        :type _discontinue: threading.Event

        :ivar fm_observability: An instance of FMObservabilityStats utilized
            for managing and tracking observability-related statistics.
        :type fm_observability: FMObservabilityStats
        """
        super().__init__()
        self._discontinue = threading.Event()
        self.fm_observability = FMObservabilityStats()

    def run(self):
        """
        Process events from the queue and handle them utilizing the observability framework.

        This method continuously polls the `_fm_observability_queue` for events. If an event is retrieved,
        it invokes the `on_event` method of the `fm_observability` instance to handle the event. Should
        the `_discontinue` flag be set, the method ceases its operation. If the queue remains empty
        for a specified timeout interval, the method continues its loop without handling an event.

        :raises queue.Empty: Raised when attempting to get an item from an empty queue and the timeout expires.
        """
        logging.debug('Starting queue poller')
        while not self._discontinue.is_set():
            try:
                event = _fm_observability_queue.get(timeout=1)
                if event:
                    self.fm_observability.on_event(event=event)
            except queue.Empty:
                pass

    def stop(self):
        """
        Stops the queue polling process by setting the internal flag that signals
        polling discontinuation. This method is intended to halt any ongoing
        polling activities cleanly and efficiently. It logs the stopping operation
        and provides access to relevant observability metrics upon completion.

        :return: Observability metrics related to the queue polling process.
        :rtype: Any
        """
        logging.debug('Stopping queue poller')
        self._discontinue.set()
        return self.fm_observability


@dataclass
class FMObservabilityStats:
    """
    Represents statistics for tracking aggregated performance metrics for
    language model (LLM) operations and embedding operations.

    This class is designed for the purpose of collecting and maintaining
    performance metrics such as execution durations, counts, and token usage
    for language model operations and embeddings. It provides methods for
    updating the statistics based on incoming data and properties to
    calculate derived statistics such as averages. This can be useful for
    monitoring and analyzing the behavior and resource utilization of large
    language models and embedding processes.

    :ivar total_llm_duration_millis: Cumulative total duration in milliseconds
        of all LLM operations.
    :type total_llm_duration_millis: float
    :ivar total_llm_count: Total number of LLM operations executed.
    :type total_llm_count: int
    :ivar total_llm_prompt_tokens: Total number of tokens used in prompts
        across all LLM operations.
    :type total_llm_prompt_tokens: float
    :ivar total_llm_completion_tokens: Total number of tokens used in
        completions across all LLM operations.
    :type total_llm_completion_tokens: float
    :ivar total_embedding_duration_millis: Cumulative total duration in
        milliseconds of all embedding operations.
    :type total_embedding_duration_millis: float
    :ivar total_embedding_count: Total number of embedding operations executed.
    :type total_embedding_count: int
    :ivar total_embedding_tokens: Total number of tokens used in all embedding
        operations.
    :type total_embedding_tokens: float
    """

    total_llm_duration_millis: float = 0
    total_llm_count: int = 0
    total_llm_prompt_tokens: float = 0
    total_llm_completion_tokens: float = 0
    total_embedding_duration_millis: float = 0
    total_embedding_count: int = 0
    total_embedding_tokens: float = 0

    def update(self, stats: Any):
        """
        Updates the current object's cumulative statistics based on another statistics object
        and determines if there are any recorded LLM or embedding operations.

        This method aggregates various metrics related to LLM (Large Language Model) operations
        and embeddings, including their respective durations, counts, and token usage.

        :param stats: The statistics object containing the metrics to be added to the current object.
        :type stats: Any
        :return: A boolean indicating whether there was at least one recorded LLM or embedding operation.
        :rtype: bool
        """
        self.total_llm_duration_millis += stats.total_llm_duration_millis
        self.total_llm_count += stats.total_llm_count
        self.total_llm_prompt_tokens += stats.total_llm_prompt_tokens
        self.total_llm_completion_tokens += stats.total_llm_completion_tokens
        self.total_embedding_duration_millis += stats.total_embedding_duration_millis
        self.total_embedding_count += stats.total_embedding_count
        self.total_embedding_tokens += stats.total_embedding_tokens
        return (stats.total_llm_count + stats.total_embedding_count) > 0

    def on_event(self, event: CBEvent):
        """
        Handles the processing of events with different types, updating various counters
        and accumulators based on the event payload. The function is invoked whenever an
        event occurs, and modifies the object's state based on the event type and its
        associated payload data.

        :param event: A CBEvent instance representing the event to handle. It contains
            details including the event type and a payload dict that holds related information.
        :type event: CBEvent

        :return: None
        """
        if event.event_type == CBEventType.LLM:
            if 'model' in event.payload:
                self.total_llm_duration_millis += event.payload['duration_millis']
                self.total_llm_count += 1
            elif 'llm_prompt_token_count' in event.payload:
                self.total_llm_prompt_tokens += event.payload['llm_prompt_token_count']
                self.total_llm_completion_tokens += event.payload[
                    'llm_completion_token_count'
                ]
        elif event.event_type == CBEventType.EMBEDDING:
            if 'model' in event.payload:
                self.total_embedding_duration_millis += event.payload['duration_millis']
                self.total_embedding_count += 1
            elif 'embedding_token_count' in event.payload:
                self.total_embedding_tokens += event.payload['embedding_token_count']

    @property
    def average_llm_duration_millis(self) -> int:
        """
        Provides the average duration of LLM operations in milliseconds.

        This property calculates the average duration based on the total LLM
        duration and count. If there have been no LLM operations recorded
        (total_llm_count is 0), it returns 0 to avoid division by zero.

        :return: The average LLM duration in milliseconds. If there are
            no LLM operations recorded, it returns 0.
        :rtype: int
        """
        if self.total_llm_count > 0:
            return self.total_llm_duration_millis / self.total_llm_count
        else:
            return 0

    @property
    def total_llm_tokens(self) -> int:
        """
        Calculates the total number of tokens used in an LLM operation.

        The total includes both the prompt tokens and the completion tokens, which
        are summed to determine the overall usage.

        :return: Total number of tokens used in an LLM operation.
        :rtype: int
        """
        return self.total_llm_prompt_tokens + self.total_llm_completion_tokens

    @property
    def average_llm_prompt_tokens(self) -> int:
        """
        Computes the average number of tokens in LLM (Large Language Model) prompts.

        This property calculates the average by dividing the total number of tokens in
        LLM prompts (`total_llm_prompt_tokens`) by the total count of LLMs (`total_llm_count`).
        If the total count of LLMs is zero, the result will be zero to avoid division
        errors and maintain a sensible return value. This property ensures easy and
        efficient access to the token average used across LLM prompts.

        :return: The average number of tokens in LLM prompts as an integer. Defaults
                 to zero if no LLM data is available.
        :rtype: int
        """
        if self.total_llm_count > 0:
            return self.total_llm_prompt_tokens / self.total_llm_count
        else:
            return 0

    @property
    def average_llm_completion_tokens(self) -> int:
        """
        Computes the average number of tokens used in the completion phase of
        LLM (Large Language Model) interactions. This metric is calculated by dividing
        the total completion tokens by the total count of LLM interactions. If no LLM
        interactions have occurred, the function returns 0 to avoid division by zero.

        :return: The average number of completion tokens per LLM interaction.
        :rtype: int
        """
        if self.total_llm_count > 0:
            return self.total_llm_completion_tokens / self.total_llm_count
        else:
            return 0

    @property
    def average_llm_tokens(self) -> int:
        """
        Calculates the average number of Large Language Model (LLM) tokens per
        count. If the total count of LLM instances is greater than zero, the method
        returns the total tokens divided by the total count. Otherwise, it returns zero.

        :return: The average number of LLM tokens as an integer.
        :rtype: int
        """
        if self.total_llm_count > 0:
            return self.total_llm_tokens / self.total_llm_count
        else:
            return 0

    @property
    def average_embedding_duration_millis(self) -> int:
        """
        Calculates and returns the average embedding duration in milliseconds.

        The calculation is based on the total embedding duration and the total number
        of embeddings. If there are no embeddings, the value returned is 0. This
        property provides a convenient way to access the average embedding duration
        directly.

        :return: The average embedding duration in milliseconds. Returns 0 if there
            are no embeddings.
        :rtype: int
        """
        if self.total_embedding_count > 0:
            return self.total_embedding_duration_millis / self.total_embedding_count
        else:
            return 0

    @property
    def average_embedding_tokens(self) -> int:
        """
        Calculates and returns the average number of embedding tokens. The average
        is computed by dividing the total number of embedding tokens by the total
        embedding count. If the total embedding count is zero, the method returns
        zero to prevent division by zero.

        :return: Integer value representing the average number of embedding tokens.
                 Returns 0 if the total embedding count is zero.
        :rtype: int
        """
        if self.total_embedding_count > 0:
            return self.total_embedding_tokens / self.total_embedding_count
        else:
            return 0


class FMObservabilitySubscriber(ABC):
    """
    Interface defining an observability subscriber mechanism.

    The FMObservabilitySubscriber class is an abstract base class intended to be
    used as a blueprint for creating subscribers that respond to updates in
    observability data. Concrete implementations are expected to define the behavior
    required for processing newly received statistics. This class enforces the
    implementation of the `on_new_stats` method, ensuring that it processes
    instances of FMObservabilityStats.

    """

    @abstractmethod
    def on_new_stats(self, stats: FMObservabilityStats):
        """
        Abstract function that must be implemented by a derived class to process and
        handle new FMObservabilityStats statistics. This method is expected to define
        specific actions to be performed whenever a new instance of FMObservabilityStats
        is received.

        :param stats: The new FMObservabilityStats object containing the statistics data
                      to be handled. The exact nature of the data will depend on the
                      implementation and definition of FMObservabilityStats.

        """
        pass


class ConsoleFMObservabilitySubscriber(FMObservabilitySubscriber):
    """
    Subscriber for managing and observing statistics within a console-based environment.

    This class implements methods for handling and updating observability statistics
    for foundational models (FM). The `ConsoleFMObservabilitySubscriber` maintains
    an aggregated view of statistics and provides a mechanism to output these
    statistics directly to the console for real-time monitoring.

    :ivar all_stats: Instance of `FMObservabilityStats` that aggregates and manages
        all statistics related to observability.
    :type all_stats: FMObservabilityStats
    """

    def __init__(self):
        """
        The class is responsible for managing and tracking the statistics of observability data, specifically focusing
        on FM (fault management) metrics. It provides a central location for storing and accessing aggregated statistical
        information within the context of monitoring systems. This class initializes with an instance of
        FMObservabilityStats.

        Attributes
        ----------
        all_stats : FMObservabilityStats
            An object responsible for collecting and maintaining fault management observability metrics.
        """
        self.all_stats = FMObservabilityStats()

    def on_new_stats(self, stats: FMObservabilityStats):
        """
        Updates the current statistics with new observations and logs detailed information
        about LLM-related and embeddings-related counts and tokens if the statistics are
        updated successfully.

        :param stats: The new set of statistics to be merged into the existing stats.
        :type stats: FMObservabilityStats
        :return: None
        """
        updated = self.all_stats.update(stats)
        if updated:
            print(
                f'LLM: count: {self.all_stats.total_llm_count}, total_prompt_tokens: {self.all_stats.total_llm_prompt_tokens}, total_completion_tokens: {self.all_stats.total_llm_completion_tokens}'
            )
            print(
                f'Embeddings: count: {self.all_stats.total_embedding_count}, total_tokens: {self.all_stats.total_embedding_tokens}'
            )


class StatPrintingSubscriber(FMObservabilitySubscriber):
    """
    Manages, tracks, and reports observability-related cost statistics for a foundation model.

    The class records cost metrics for processing tokens (input, output, and embedding)
    and aggregates usage statistics. It provides methods to estimate costs, retrieve detailed
    statistics, and update the internal state with new data. This class helps monitor and
    analyze the token and duration-based usages and compute associated costs dynamically.

    :ivar cost_per_thousand_input_tokens_llm: Cost associated with processing one thousand input tokens by the language model.
    :type cost_per_thousand_input_tokens_llm: float
    :ivar cost_per_thousand_output_tokens_llm: Cost associated with processing one thousand output tokens by the language model.
    :type cost_per_thousand_output_tokens_llm: float
    :ivar cost_per_thousand_embedding_tokens: Cost associated with processing one thousand embedding tokens.
    :type cost_per_thousand_embedding_tokens: float
    :ivar all_stats: Aggregated statistics tracking object for tokens, durations, and usage metrics.
    :type all_stats: FMObservabilityStats
    """

    cost_per_thousand_input_tokens_llm: float = 0
    cost_per_thousand_output_tokens_llm: float = 0
    cost_per_thousand_embedding_tokens: float = 0

    def __init__(
        self,
        cost_per_thousand_input_tokens_llm,
        cost_per_thousand_output_tokens_llm,
        cost_per_thousand_embedding_tokens,
    ):
        """
        Initializes an instance to manage and track observability-related cost statistics for a
        foundation model including costs for input tokens, output tokens, and embedding tokens.
        The class encapsulates these parameters as input cost metrics and initializes an
        aggregated statistics tracking object.

        :param cost_per_thousand_input_tokens_llm:
            Cost associated with processing one thousand input tokens by the language model.
        :type cost_per_thousand_input_tokens_llm: float
        :param cost_per_thousand_output_tokens_llm:
            Cost associated with processing one thousand output tokens by the language model.
        :type cost_per_thousand_output_tokens_llm: float
        :param cost_per_thousand_embedding_tokens:
            Cost associated with processing one thousand embedding tokens.
        :type cost_per_thousand_embedding_tokens: float
        """
        self.all_stats = FMObservabilityStats()
        self.cost_per_thousand_input_tokens_llm = cost_per_thousand_input_tokens_llm
        self.cost_per_thousand_output_tokens_llm = cost_per_thousand_output_tokens_llm
        self.cost_per_thousand_embedding_tokens = cost_per_thousand_embedding_tokens

    def on_new_stats(self, stats: FMObservabilityStats):
        """
        Processes and updates internal statistics with the provided new data.

        This method takes an instance of `FMObservabilityStats` containing new
        observability statistics and integrates it into the current set of
        stored statistics by updating the internal state of `all_stats`.

        :param stats: New observation statistics to integrate.
        :type stats: FMObservabilityStats

        """
        self.all_stats.update(stats)

    def get_stats(self):
        """
        Provides a method to retrieve statistical data stored in the instance.

        This method is used to access the `all_stats` attribute, which contains
        the aggregated statistical data collected and managed within the instance.

        :return: The aggregated statistical data.
        :rtype: Any
        """
        return self.all_stats

    def estimate_costs(self) -> float:
        """
        Estimate the total cost based on token usage and predefined costs per thousand tokens.
        The function calculates the cost using prompt tokens, completion tokens,
        and embedding tokens, each scaled by their respective costs.

        :return: Total calculated cost as a float.
        :rtype: float
        """
        total_cost = (
            self.all_stats.total_llm_prompt_tokens
            / 1000.0
            * self.cost_per_thousand_input_tokens_llm
            + self.all_stats.total_llm_completion_tokens
            / 1000.0
            * self.cost_per_thousand_output_tokens_llm
            + self.all_stats.total_embedding_tokens
            / 1000.0
            * self.cost_per_thousand_embedding_tokens
        )
        return total_cost

    def return_stats_dict(self) -> Dict[str, Any]:
        """
        Generates and returns a dictionary containing various statistical data, including
        counts, tokens, durations, and estimated costs. The data is aggregated from a
        set of statistics managed within the class instance.

        :return: A dictionary containing statistical information.
        :rtype: Dict[str, Any]
        """
        stats_dict = {}
        stats_dict['total_llm_count'] = self.all_stats.total_llm_count
        stats_dict['total_prompt_tokens'] = self.all_stats.total_llm_prompt_tokens
        stats_dict['total_completion_tokens'] = (
            self.all_stats.total_llm_completion_tokens
        )
        # Now embeddings count and total embedding tokens
        stats_dict['total_embedding_count'] = self.all_stats.total_embedding_count
        stats_dict['total_embedding_tokens'] = self.all_stats.total_embedding_tokens
        # Now duration data
        stats_dict["total_llm_duration_millis"] = (
            self.all_stats.total_llm_duration_millis
        )
        stats_dict["total_embedding_duration_millis"] = (
            self.all_stats.total_embedding_duration_millis
        )
        stats_dict["average_llm_duration_millis"] = (
            self.all_stats.average_llm_duration_millis
        )
        stats_dict["average_embedding_duration_millis"] = (
            self.all_stats.average_embedding_duration_millis
        )
        # Now  costs
        stats_dict['total_llm_cost'] = self.estimate_costs()
        return stats_dict


class FMObservabilityPublisher:
    """
    Manages a list of subscribers to periodically publish collected statistics
    using a polling mechanism, and provides lifecycle management capabilities.

    This class encapsulates the functionality needed for subscribing to
    observability events, collecting related statistics periodically, and handling
    publishing these statistics to the subscribers. It also supports integration
    with context management for automated resource cleanup.

    :ivar subscribers: List of subscribers to receive published statistics.
        Subscribers must implement the FMObservabilitySubscriber interface.
    :type subscribers: List[FMObservabilitySubscriber]
    :ivar interval_seconds: Configurable time interval (in seconds) for publishing
        statistics periodically.
    :type interval_seconds: float
    :ivar allow_continue: Flag to indicate if the continuous publishing process
        is allowed or should be halted.
    :type allow_continue: bool
    :ivar poller: Instance of FMObservabilityQueuePoller that is responsible for
        gathering statistical data.
    :type poller: FMObservabilityQueuePoller
    """

    def __init__(
        self, subscribers: List[FMObservabilitySubscriber] = [], interval_seconds=15.0
    ):
        """
        Initializes FMObservability, a system for managing observability with subscribers
        and periodic publishing of statistics.

        The constructor sets up a global observability queue, binds necessary handlers
        to the callback manager, initializes subscribers, the publishing interval, and
        a poller. It also starts a timer for periodic publishing of metrics or statistics.

        :param subscribers: List of FMObservabilitySubscriber objects that will handle
            observability data.
        :param interval_seconds: The interval in seconds at which stats will be published.
        :type interval_seconds: float
        """
        global _fm_observability_queue
        _fm_observability_queue = Queue()

        Settings.callback_manager.add_handler(BedrockEnabledTokenCountingHandler())
        Settings.callback_manager.add_handler(FMObservabilityHandler())

        self.subscribers = subscribers
        self.interval_seconds = interval_seconds
        self.allow_continue = True
        self.poller = FMObservabilityQueuePoller()
        self.poller.start()

        threading.Timer(interval_seconds, self.publish_stats).start()

    def close(self):
        """
        Disables the continuation state of the instance by setting the `allow_continue`
        attribute to False. This method typically halts processes or terminates
        iterations controlled by the `allow_continue` flag.

        :return: None
        """
        self.allow_continue = False

    def __enter__(self):
        """
        Provides functionality for managing initialization of a context in a
        with-statement block. This method is invoked upon entering the runtime
        context.

        :return: The instance of the object itself that implements the context
                 manager protocol.
        :rtype: self
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Handles the context manager exit functionality for a resource by ensuring
        that it gets properly closed. This method is invoked when exiting the
        runtime context related to the resource managed by the context manager.

        :param exc_type: The exception type, if an exception occurred while
            using the resource. Otherwise, it will be None.
        :param exc_value: The exception instance, if an exception occurred while
            using the resource. Otherwise, it will be None.
        :param exc_traceback: The traceback object, if an exception occurred while
            using the resource. Otherwise, it will be None.
        :return: Always returns None to indicate that any exception
            should be re-raised after cleanup.
        """
        self.close()

    def publish_stats(self):
        """
        Stop the current statistics polling, restart the poller, and dispatch new statistics
        to all subscribers. The method schedules another polling cycle if allowed.

        :raises: None

        :param None

        :return: None
        """
        stats = self.poller.stop()
        self.poller = FMObservabilityQueuePoller()
        self.poller.start()
        if self.allow_continue:
            logging.debug('Scheduling new poller')
            threading.Timer(self.interval_seconds, self.publish_stats).start()
        else:
            logging.debug('Shutting down publisher')
        for subscriber in self.subscribers:
            subscriber.on_new_stats(stats)


def get_patched_llm_token_counts(
    token_counter: TokenCounter, payload: Dict[str, Any], event_id: str = ""
) -> TokenCountingEvent:
    """
    Calculates and returns token counts for a given payload using a provided token counter.
    It processes payloads that contain either a prompt and completion, or messages and a
    response. Additionally, it attempts to retrieve token counts from pre-attached metadata
    or computes the counts if metadata is unavailable or invalid.

    :param token_counter: A utility for counting tokens in text or message objects.
    :type token_counter: TokenCounter
    :param payload: A dictionary containing the details of the LLM request/response. The
        payload is expected to have either a `PROMPT` and `COMPLETION` or
        `MESSAGES` and `RESPONSE` key.
    :type payload: Dict[str, Any]
    :param event_id: An optional unique identifier for the token counting event. Defaults
        to an empty string if not provided.
    :type event_id: str
    :raises ValueError: If the payload does not contain required keys or contains invalid
        token counts.
    :return: A TokenCountingEvent containing details about the token counts for prompt
        and completion strings/messages.
    :rtype: TokenCountingEvent
    """
    from llama_index.core.llms import ChatMessage

    if EventPayload.PROMPT in payload:
        prompt = str(payload.get(EventPayload.PROMPT))
        completion = str(payload.get(EventPayload.COMPLETION))

        return TokenCountingEvent(
            event_id=event_id,
            prompt=prompt,
            prompt_token_count=token_counter.get_string_tokens(prompt),
            completion=completion,
            completion_token_count=token_counter.get_string_tokens(completion),
        )

    elif EventPayload.MESSAGES in payload:
        messages = cast(List[ChatMessage], payload.get(EventPayload.MESSAGES, []))
        messages_str = "\n".join([str(x) for x in messages])

        response = payload.get(EventPayload.RESPONSE)
        response_str = str(response)

        # try getting attached token counts first
        try:
            messages_tokens = 0
            response_tokens = 0

            if response is not None and response.raw is not None:
                usage = response.raw.get("usage", None)

                if usage is not None:
                    if not isinstance(usage, dict):
                        usage = dict(usage)
                    messages_tokens = usage.get(
                        "prompt_tokens", usage.get("input_tokens", 0)
                    )
                    response_tokens = usage.get(
                        "completion_tokens", usage.get("output_tokens", 0)
                    )

                if messages_tokens == 0 or response_tokens == 0:
                    raise ValueError("Invalid token counts!")

                return TokenCountingEvent(
                    event_id=event_id,
                    prompt=messages_str,
                    prompt_token_count=messages_tokens,
                    completion=response_str,
                    completion_token_count=response_tokens,
                )

        except (ValueError, KeyError):
            # Invalid token counts, or no token counts attached
            pass

        # Should count tokens ourselves
        messages_tokens = token_counter.estimate_tokens_in_messages(messages)
        response_tokens = token_counter.get_string_tokens(response_str)

        return TokenCountingEvent(
            event_id=event_id,
            prompt=messages_str,
            prompt_token_count=messages_tokens,
            completion=response_str,
            completion_token_count=response_tokens,
        )
    else:
        raise ValueError(
            "Invalid payload! Need prompt and completion or messages and response."
        )


class BedrockEnabledTokenCountingHandler(TokenCountingHandler):
    """
    Handles token counting and observability queue management in the context of
    callbacks for specific events. It extends the TokenCountingHandler to provide
    additional functionality tailored for use cases requiring direct association
    with Bedrock-enabling mechanisms.

    The class manages token counting for specific event types (e.g., LLM and
    Embedding events), allows the exclusion of certain events from processing, and
    ensures that token count lists are cleared when they exceed a threshold. It
    also integrates with an observability queue for structured monitoring and
    debugging purposes.

    :ivar tokenizer: Callable function responsible for tokenizing input strings
        into a list of tokens. Defaults to None if not explicitly set.
    :type tokenizer: Optional[Callable[[str], List]]
    :ivar event_starts_to_ignore: List of event types to be ignored during
        start event operations. This allows for filtering out unneeded
        processing. Defaults to None if not explicitly set.
    :type event_starts_to_ignore: Optional[List[CBEventType]]
    :ivar event_ends_to_ignore: List of event types to be ignored during
        end event operations. This provides a mechanism for tailoring token
        counting behavior. Defaults to None if not explicitly set.
    :type event_ends_to_ignore: Optional[List[CBEventType]]
    :ivar verbose: A boolean flag determining whether verbose output is enabled.
        This helps in logging and debugging by providing more detailed system
        behavior information.
    :type verbose: bool
    :ivar logger: Logging instance for capturing debug, info, or error logs
        during operation. If not provided, logging will be disabled.
    :type logger: Optional[logging.Logger]
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List]] = None,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the class with the specified parameters to customize behavior,
        particularly for handling tokenization and event filtering during execution.

        :param tokenizer: A callable that takes a string input and tokenizes it into a list.
            Default is None.
        :param event_starts_to_ignore: A list of event start types (`CBEventType`) to ignore.
            Default is None.
        :param event_ends_to_ignore: A list of event end types (`CBEventType`) to ignore.
            Default is None.
        :param verbose: A boolean flag to enable or disable verbose output. Default is False.
        :param logger: An optional logger instance for logging messages. Default is None.
        """
        import llama_index.core.callbacks.token_counting

        llama_index.core.callbacks.token_counting.get_llm_token_counts = (
            get_patched_llm_token_counts
        )

        super().__init__(
            tokenizer=tokenizer,
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
            verbose=verbose,
            logger=logger,
        )

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Handles the ending of an event, processes event payloads, and updates observability
        queues if necessary. This method primarily processes LLM and EMBEDDING event types,
        extracts relevant token count information, and dispatches them to the observability
        queue. It also ensures that token count tracking is reset if their sizes exceed
        the threshold limit.

        :param event_type: The type of the event being ended (e.g., LLM, EMBEDDING).
        :param payload: The payload data associated with the event, including tokens
                        or other metadata. If None, no payload will be processed.
        :param event_id: The unique identifier for the event. Useful for tracking specific
                         events across the system.
        :param kwargs: Any additional named parameters required for custom handling.
        :return: None
        """
        super().on_event_end(event_type, payload, event_id, **kwargs)

        event_payload = None
        """Count the LLM or Embedding tokens as needed."""
        if (
            event_type == CBEventType.LLM
            and event_type not in self.event_ends_to_ignore
            and payload is not None
        ):
            event_payload = {
                'llm_prompt_token_count': self.llm_token_counts[-1].prompt_token_count,
                'llm_completion_token_count': self.llm_token_counts[
                    -1
                ].completion_token_count,
            }
        elif (
            event_type == CBEventType.EMBEDDING
            and event_type not in self.event_ends_to_ignore
            and payload is not None
        ):
            event_payload = {
                'embedding_token_count': self.embedding_token_counts[
                    -1
                ].total_token_count
            }

        if event_payload:

            event = CBEvent(event_type=event_type, payload=event_payload, id_=event_id)

            _fm_observability_queue.put(event)

        if len(self.llm_token_counts) > 1000 or len(self.embedding_token_counts) > 1000:
            self.reset_counts()


class FMObservabilityHandler(BaseCallbackHandler):
    """
    Handles observability for monitored events by managing in-flight event tracking and
    trace sessions.

    This class extends the functionality of BaseCallbackHandler to provide specific
    observability features for handling events. It keeps track of in-flight events, computes
    metrics such as duration for completed events, and integrates with a queuing mechanism
    for further processing. Additionally, it enables tracing sessions to facilitate monitoring
    and debugging in distributed systems.

    :ivar in_flight_events: Dictionary for tracking ongoing events with event info.
    :type in_flight_events: dict
    """

    def __init__(self, event_starts_to_ignore=[], event_ends_to_ignore=[]):
        """
        Represents an event tracker used to handle in-flight events and manage ignored
        start and end events. This class is designed to store and track details
        of ongoing events and contains the logic for keeping a record of events
        specified under the categories of ignored starts or ends.

        This class initializes in-flight events and allows for the management
        or extension of event tracking mechanisms.

        :param event_starts_to_ignore: A list of event names or identifiers to be
            ignored during the event start tracking. Defaults to an empty list.
        :type event_starts_to_ignore: list
        :param event_ends_to_ignore: A list of event names or identifiers to be
            ignored during the event end tracking. Defaults to an empty list.
        :type event_ends_to_ignore: list
        """
        super().__init__(event_starts_to_ignore, event_ends_to_ignore)
        self.in_flight_events = {}

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Handles the initiation of a start event by processing its associated payload
        and capturing relevant details for in-flight tracking. It checks the event
        type and payload, processes specific event data based on type, and stores the
        event in a tracking dictionary.

        :param event_type: The type of event being started.
        :type event_type: CBEventType
        :param payload: Optional dictionary containing additional details about the
            event, may hold serialized data for specific event types.
        :type payload: Optional[Dict[str, Any]]
        :param event_id: A unique identifier for the event.
        :type event_id: str
        :param parent_id: An identifier indicating the parent event, if applicable.
        :type parent_id: str
        :param kwargs: Additional arbitrary keyword arguments for flexibility.
        :type kwargs: Any
        :return: The event ID of the initiated event.
        :rtype: str
        """
        if event_type not in self.event_ends_to_ignore and payload is not None:
            if (event_type == CBEventType.LLM and EventPayload.MESSAGES in payload) or (
                event_type == CBEventType.EMBEDDING
                and EventPayload.SERIALIZED in payload
            ):
                serialized = payload.get(EventPayload.SERIALIZED, {})
                ms = time.time_ns() // 1_000_000
                event_payload = {
                    'model': serialized.get(
                        'model', serialized.get('model_name', 'unknown')
                    ),
                    'start': ms,
                }

                self.in_flight_events[event_id] = CBEvent(
                    event_type=event_type, payload=event_payload, id_=event_id
                )
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Handles the termination of an event and updates its metrics such as duration.
        The function processes the event by checking whether it should be ignored and
        ensures the appropriate payload keys are present based on the event type. If
        the event is valid, its duration is calculated in milliseconds and the event
        is staged for further processing by adding it to the observability queue.

        :param event_type: The type of the event. Must be of type CBEventType.
        :param payload: The dictionary containing additional data about the event.
            It is optional and defaults to None.
        :param event_id: A unique identifier for the event. Defaults to an empty
            string if not provided.
        :param kwargs: Additional optional keyword arguments.
        :return: None
        """
        if event_type not in self.event_ends_to_ignore and payload is not None:
            if (event_type == CBEventType.LLM and EventPayload.MESSAGES in payload) or (
                event_type == CBEventType.EMBEDDING
                and EventPayload.EMBEDDINGS in payload
            ):
                try:
                    event = self.in_flight_events.pop(event_id)

                    start_ms = event.payload['start']
                    end_ms = time.time_ns() // 1_000_000
                    event.payload['duration_millis'] = end_ms - start_ms

                    _fm_observability_queue.put(event)
                except KeyError:
                    pass

    def reset_counts(self) -> None:
        """
        Resets the in-flight events dictionary to an empty state.

        This method clears the `in_flight_events` attribute, which keeps track of
        the events currently being processed. It resets it back to an empty dictionary,
        effectively clearing all tracked in-flight events.

        :return: None. The method does not return any value.
        """
        self.in_flight_events = {}

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """
        Starts a trace session, initializing it with the provided trace ID or generating
        a new one if none is supplied. This is typically used to track operations or
        activities across a distributed system or within an application for debugging
        and monitoring purposes.

        :param trace_id: Optional; The trace ID to be used for this trace session. If
            omitted, the system will generate a new trace ID.
        :return: None
        """
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Ends the trace with the provided `trace_id` and cleans up the related
        mapping from `trace_map`, if applicable. Used to signify the completion
        of a trace session and optionally to maintain the integrity of a
        trace-tracking system.

        :param trace_id: An optional identifier for the trace to be ended. If not provided,
            it assumes no specific trace ID is targeted.
        :param trace_map: An optional dictionary of trace mappings where the trace ID
            could be associated with one or multiple session identifiers. This structure
            helps maintain a reference and cleanup of related associations.
        :return: Nothing
        """
        pass
