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
    Handles queue polling for observability purposes on a separate thread.

    This class extends `threading.Thread` to perform continuous polling of a
    queue that collects observability events. It processes these events by
    delegating them to the `FMObservabilityStats` instance. The class also
    provides mechanisms to start and stop the polling process safely using
    thread signaling.

    :ivar _discontinue: Threading event used to signal and manage thread lifecycle
        or operations that need coordination.
    :type _discontinue: threading.Event
    :ivar fm_observability: Instance of FMObservabilityStats that handles
        observability-related statistics and functionality.
    :type fm_observability: FMObservabilityStats
    """

    def __init__(self):
        """
        Represents the initialization of a class that manages threading events and
        observability statistics. The constructor sets up an event object for signaling
        and an instance of an observability statistics tracker.

        Attributes:
            _discontinue (threading.Event): A threading event used to signal and manage
                lifecycle events or state changes within the context of threading.
                It allows threads to communicate and signal state changes effectively.
            fm_observability (FMObservabilityStats): An instance of the
                FMObservabilityStats class, responsible for tracking and handling
                observability-related statistics or data points.

        Methods:
            This class only contains initializations and does not define other methods
            explicitly. Method-specific details are not documented here as this docstring
            pertains to the class-level overview.
        """
        super().__init__()
        self._discontinue = threading.Event()
        self.fm_observability = FMObservabilityStats()

    def run(self):
        """
        Polls a queue to process incoming events and passes them to an observer. The method
        continuously runs in a loop until a stopping signal is triggered externally. If an event
        is retrieved from the queue, it is processed using the `fm_observability` attribute.

        :raises queue.Empty: when the queue is empty and no event is found within the timeout
            period.
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
        Stops the queue poller and updates observability status.

        The method halts the queue polling process by setting an internal flag,
        `_discontinue`, which signals the process to stop. It then returns the
        observability status for further evaluation or logging.

        :param self: The instance of the class where the method is called.

        :return: Observability status after stopping the queue poller. The type
            of the return value corresponds to the `fm_observability` attribute.
        """
        logging.debug('Stopping queue poller')
        self._discontinue.set()
        return self.fm_observability


@dataclass
class FMObservabilityStats:
    """
    Encapsulates observability statistics for operations involving Language
    Learning Models (LLM) and embeddings.

    This class is designed to track and manage cumulative statistics related to
    LLM and embedding operations, including their duration, count, and token usage.
    It includes methods for updating these statistics from external inputs,
    handling events, and computing various metrics such as averages for time
    and token usage. This class can be used in scenarios where monitoring and
    analysis of LLM and embedding system performance is required.

    :ivar total_llm_duration_millis: Total duration of all LLM operations in milliseconds.
    :type total_llm_duration_millis: float
    :ivar total_llm_count: Total number of LLM operations conducted.
    :type total_llm_count: int
    :ivar total_llm_prompt_tokens: Cumulative number of tokens used in prompts for LLM.
    :type total_llm_prompt_tokens: float
    :ivar total_llm_completion_tokens: Cumulative number of tokens generated in the
        completion phase by the LLM.
    :type total_llm_completion_tokens: float
    :ivar total_embedding_duration_millis: Total duration of all embedding operations in milliseconds.
    :type total_embedding_duration_millis: float
    :ivar total_embedding_count: Total number of embedding operations conducted.
    :type total_embedding_count: int
    :ivar total_embedding_tokens: Total number of tokens used in all embedding operations.
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
        Updates the total statistics with values from another statistics object by
        adding the individual components of both. The method accumulates the total
        durations, counts, and token values for language model operations and
        embedding operations. Returns a boolean indicating whether the combined total
        count of operations (language model and embeddings) is greater than zero.

        :param stats: The statistics object whose values are added to the current
            object.
        :type stats: Any
        :return: A boolean value indicating if the combined total count of language
            model operations and embedding operations is greater than zero.
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
        Handles processing of incoming events and updates the relevant metrics based
        on the type and payload of the event. Processed event types include language
        model (LLM) and embedding events, where specific values are extracted from the
        event payload and used to increment counters and accumulate durations or token
        counts as necessary.

        :param event: The event instance containing information about type of event
            (e.g., LLM or embedding) and associated payload with relevant metrics data
            such as duration or token counts.
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
        Calculates and returns the average duration of LLM operations in
        milliseconds. The calculation is performed by dividing the total
        LLM duration in milliseconds by the total count of LLM operations.
        If there are no LLM operations performed, the function returns 0.

        :return: The average duration of LLM operations in milliseconds.
        :rtype: int
        """
        if self.total_llm_count > 0:
            return self.total_llm_duration_millis / self.total_llm_count
        else:
            return 0

    @property
    def total_llm_tokens(self) -> int:
        """
        Computes the total number of tokens utilized by the language model by summing
        the prompt tokens and completion tokens. This property encapsulates the logic
        to calculate the combined token usage dynamically.

        :return: The combined total of prompt and completion tokens used by
                 the language model.
        :rtype: int
        """
        return self.total_llm_prompt_tokens + self.total_llm_completion_tokens

    @property
    def average_llm_prompt_tokens(self) -> int:
        """
        Calculates and returns the average number of tokens used in LLM (Large Language
        Model) prompts. This is a derived property that computes the average based on
        the total token count in prompts and the total number of LLM executions.

        :returns:
            The average number of LLM prompt tokens as an integer. Returns 0 if the
            total LLM count is 0.
        """
        if self.total_llm_count > 0:
            return self.total_llm_prompt_tokens / self.total_llm_count
        else:
            return 0

    @property
    def average_llm_completion_tokens(self) -> int:
        """
        Represents the average number of LLM completion tokens calculated
        based on the total number of LLM completion tokens and the total
        LLM count. Returns zero if the total LLM count is zero.

        :return: The average number of LLM completion tokens as an integer.
        :rtype: int
        """
        if self.total_llm_count > 0:
            return self.total_llm_completion_tokens / self.total_llm_count
        else:
            return 0

    @property
    def average_llm_tokens(self) -> int:
        """
        Provides the average number of tokens used by the language model (LLM) per
        interaction. This property calculates the average using the total number of
        tokens processed by the LLM and the count of interactions if available.
        Returns zero if there are no interactions.

        :return: The average number of tokens per interaction.
        :rtype: int
        """
        if self.total_llm_count > 0:
            return self.total_llm_tokens / self.total_llm_count
        else:
            return 0

    @property
    def average_embedding_duration_millis(self) -> int:
        """
        Calculates the average embedding duration in milliseconds. This property computes
        the average duration per embedding based on the total embedding duration summed
        over all embeddings and the total embedding count.

        :return: The average embedding duration in milliseconds as an integer. If the
             total embedding count is zero, returns zero.
        :rtype: int
        """
        if self.total_embedding_count > 0:
            return self.total_embedding_duration_millis / self.total_embedding_count
        else:
            return 0

    @property
    def average_embedding_tokens(self) -> int:
        """
        Property to calculate the average number of tokens per embedding. This property computes
        the average by dividing the total number of embedding tokens by the total count of
        embeddings. If no embeddings are present, the average defaults to zero.

        :return: The average number of tokens per embedding. If no embeddings are present,
            returns 0.
        :rtype: int
        """
        if self.total_embedding_count > 0:
            return self.total_embedding_tokens / self.total_embedding_count
        else:
            return 0


class FMObservabilitySubscriber(ABC):
    """
    An abstract base class that defines the interface for observability subscribers.

    This class serves as a blueprint for implementing subscribers that react
    to new observability statistics. Any concrete subclass must implement
    the abstract method `on_new_stats`, which is invoked when new
    statistics are available.

    :ivar attribute1: Description of attribute1.
    :type attribute1: type
    :ivar attribute2: Description of attribute2.
    :type attribute2: type
    """

    @abstractmethod
    def on_new_stats(self, stats: FMObservabilityStats):
        """
        Marks an abstract method to handle the event when new statistics are
        generated or received. This method requires implementation by any
        concrete class that inherits from it.

        :param stats: Object encapsulating the statistics data to be processed.
                      The instance contains observability metrics and details.
                      Must be an instance of FMObservabilityStats.

        :return: This method does not return any value as it performs an action.

        :rtype: None
        """
        pass


class ConsoleFMObservabilitySubscriber(FMObservabilitySubscriber):
    """
    Subscribes to and processes observability statistics for feature management
    systems.

    This class handles the aggregation and representation of observability statistics
    in a console-based format. It collects incoming data, updates the aggregated
    statistics, and displays summary information regarding LLM (Large Language Model)
    activity and Embedding usage.

    :ivar all_stats: Aggregated observability statistics.
    :type all_stats: FMObservabilityStats
    """

    def __init__(self):
        """
        Represents the initialization of default attributes for observability stats.

        This constructor initializes the `all_stats` attribute which holds an
        instance of `FMObservabilityStats`. This attribute is utilized to manage
        and access all statistical data related to observability.

        Attributes:
            all_stats (FMObservabilityStats): Holds the default instance
            of the `FMObservabilityStats` class for tracking observability
            statistics.
        """
        self.all_stats = FMObservabilityStats()

    def on_new_stats(self, stats: FMObservabilityStats):
        """
        Handles the intake of new statistics and updates the existing aggregated statistics. If the statistics are successfully
        updated, it will print current values including counts and tokens for LLMs and embeddings.

        :param stats: New set of statistics to integrate with the existing data.
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
    Handles the collection, aggregation, and analysis of observability statistics
    specific to language model (LLM) and embedding usage, as well as cost estimation
    associated with these processes.

    This class is used to monitor, update, and report on critical metrics such as
    token counts, execution durations, and financial costs derived from LLM and
    embedding operations. It is designed to integrate seamlessly with observability
    data structures, enabling efficient management and reporting of relevant statistics.

    :ivar cost_per_thousand_input_tokens_llm: Cost per thousand input tokens for
        language model usage.
    :type cost_per_thousand_input_tokens_llm: float
    :ivar cost_per_thousand_output_tokens_llm: Cost per thousand output tokens
        for language model generation.
    :type cost_per_thousand_output_tokens_llm: float
    :ivar cost_per_thousand_embedding_tokens: Cost per thousand embedding tokens
        used for processing.
    :type cost_per_thousand_embedding_tokens: float
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
        Updates the existing statistics with new data provided in the `stats` parameter.

        This method takes a statistics object and merges its values into the current
        statistics set, ensuring that all the new statistical data is added to the
        existing collection.

        :param stats: The statistics object containing new data to be merged.
        :type stats: FMObservabilityStats

        :return: None
        """
        self.all_stats.update(stats)

    def get_stats(self):
        """
        Retrieves and returns the statistics data.

        :return: The statistics data contained in the `all_stats` attribute.
        :rtype: Any
        """
        return self.all_stats

    def estimate_costs(self) -> float:
        """
        Estimates the total costs based on the provided token statistics and cost rates.

        This method computes the total cost incurred based on the token consumption for
        language model input prompts, completions, and embeddings. It utilizes the
        respective cost rate per thousand tokens for each category (input, output,
        and embeddings) provided in the instance variables.

        Formula for cost computation:
            - Input Prompt Cost = total_llm_prompt_tokens / 1000 * cost_per_thousand_input_tokens_llm
            - Output Completion Cost = total_llm_completion_tokens / 1000 * cost_per_thousand_output_tokens_llm
            - Embedding Cost = total_embedding_tokens / 1000 * cost_per_thousand_embedding_tokens

        :return: The estimated total cost.
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
        Constructs and returns a dictionary containing various statistics and calculated
        values for language learning models (LLMs) and embeddings, derived from the
        global `all_stats` object. It includes token counts, duration metrics, and cost
        estimates. Provides easy access to summarized information about LLM and
        embedding usage, performance, and efficiency.

        :return: A dictionary with detailed statistics related to LLM and embeddings, including
                 total counts, durations, and cost estimations.
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
    Manages observability subscribers, processes observability queue data, and publishes statistics.

    The FMObservabilityPublisher class serves as a utility to handle the collection and publishing
    of observability data. It works by managing subscribers and periodically sending them updated
    statistics. Additionally, it allows for integration with a context manager for proper resource
    cleanup.

    :ivar subscribers: A list of subscribers to receive the observability data.
                        These subscribers must implement the FMObservabilitySubscriber interface.
    :type subscribers: List[FMObservabilitySubscriber]
    :ivar interval_seconds: The interval in seconds for publishing statistics to the subscribers.
    :type interval_seconds: float
    :ivar allow_continue: A flag indicating whether the periodic polling process is allowed
                          to continue or not.
    :type allow_continue: bool
    :ivar poller: An instance of FMObservabilityQueuePoller used to handle the
                  polling mechanism for gathering observability statistics.
    :type poller: FMObservabilityQueuePoller
    """

    def __init__(
        self, subscribers: List[FMObservabilitySubscriber] = [], interval_seconds=15.0
    ):
        """
        Initializes an FMObservability instance to manage subscribers and periodically
        publish statistics through a polling mechanism. It also sets up the necessary
        callback handlers.

        :param subscribers: List of FMObservabilitySubscriber instances to subscribe
            to the observability events.
        :param interval_seconds: Time interval in seconds for triggering the statistics
            publishing mechanism. Defaults to 15.0.
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
        Disables the continuation flag to stop further processing by
        setting `allow_continue` to `False`.

        This method is used to signal that no further continuation or
        processing should occur, often in response to a specific condition
        or as part of a controlled shutdown process. Altering the
        `allow_continue` attribute ensures that the object reflects the
        halted state and prevents operations that depend on continuation.

        :return: None
        """
        self.allow_continue = False

    def __enter__(self):
        """
        This method allows the object to be used as a context manager within
        a 'with' statement. When the with block is entered, this method is
        executed, and a reference to the object itself is returned to manage
        its resource.

        :return: The object itself to manage resources during the context.
        :rtype: self
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Handles the cleanup process during the exit from a runtime context.

        This method is designed to execute any required shutdown or cleanup
        operations when exiting a runtime context (e.g., when used in a `with`
        statement). The cleanup ensures that resources are released properly.

        :param exc_type: Exception type of the raised exception, if any.
        :type exc_type: type or None
        :param exc_value: Value of the raised exception, if any.
        :type exc_value: BaseException or None
        :param exc_traceback: Traceback object of the raised exception, if any.
        :type exc_traceback: traceback or None
        :return: Indicates whether to suppress the exception.
        :rtype: bool or None
        """
        self.close()

    def publish_stats(self):
        """
        Publishes statistics collected by the poller to all subscribers and handles the lifecycle
        of the poller, including restarting or stopping it based on the configuration.

        The function retrieves the collected stats from the current poller, restarts the
        poller and starts it to continue collecting data if allowed. It schedules the next
        execution of this method at regular intervals as configured. If allowed to continue,
        it ensures the process runs periodically. Otherwise, it logs the shutdown process.
        It also notifies all registered subscribers about the newly published stats.

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
    Get token counts for LLM interactions such as prompts, completions, messages, and responses.

    This function processes a payload containing either a prompt and completion or a series
    of chat messages with a response. Depending on the presence of specific elements in the
    payload (e.g., `EventPayload.PROMPT`, `EventPayload.COMPLETION`, `EventPayload.MESSAGES`),
    it calculates token counts either based on the raw token usage in the response or by
    estimating the number of tokens based on the content of strings/messages.

    If valid raw token usage data is provided in the payload (through the `response.raw` field),
    that data is utilized for constructing the event. Otherwise, the function falls back on
    estimating token counts using the provided `TokenCounter`.

    :param token_counter: Instance of `TokenCounter` used for counting or estimating tokens in
                          strings and chat messages.
    :param payload: Dictionary containing relevant LLM interaction data, which can include:
                    - A prompt under the key `EventPayload.PROMPT`.
                    - A completion under the key `EventPayload.COMPLETION`.
                    - Chat messages under the key `EventPayload.MESSAGES`.
                    - A response under the key `EventPayload.RESPONSE`.
    :param event_id: A unique identifier for the event. Defaults to an empty string.

    :return: An instance of `TokenCountingEvent` containing details about the LLM interaction,
             including the relevant prompt, completion, and their respective token counts.
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
    Handles advanced token counting for LLM and embedding operations, equipped
    with special handling features and event queueing for observability. This
    class extends `TokenCountingHandler` by including additional logic specific
    to event-based token tracking and threshold-based resets of token counts.

    :ivar tokenizer: Callable responsible for tokenizing input strings into a list
        of tokens.
    :type tokenizer: Optional[Callable[[str], List]]
    :ivar event_starts_to_ignore: List of CBEventType items representing event
        types to ignore when a start event occurs.
    :type event_starts_to_ignore: Optional[List[CBEventType]]
    :ivar event_ends_to_ignore: List of CBEventType items representing event types
        to ignore when an end event occurs.
    :type event_ends_to_ignore: Optional[List[CBEventType]]
    :ivar verbose: Indicates whether to enable verbose output.
    :type verbose: bool
    :ivar logger: Logger instance used for logging callback-related information.
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
        """Initializes the class with specified parameters. The provided
        parameters will control the behavior of the callback system for token
        counting. This includes the tokenizer to be used, events to be ignored
        during start and end operations, verbosity level, and an optional
        logger for logging purposes.

        Args:
            tokenizer: Optional callable responsible for tokenizing input strings into
                a list of tokens.
            event_starts_to_ignore: Optional list of CBEventType items representing
                event types to ignore when a start event occurs.
            event_ends_to_ignore: Optional list of CBEventType items representing event
                types to ignore when an end event occurs.
            verbose: Boolean indicating whether to enable verbose output.
            logger: Optional logging.Logger instance for logging callback-related
                information.
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
        Handles the end of an event by processing token counts for LLM or Embedding
        events, and adding them to the observability queue if applicable. It also
        ensures token count lists are cleared when they exceed a threshold.

        :param event_type: The type of the event being handled.
        :type event_type: CBEventType
        :param payload: Additional event-specific data. Defaults to None.
        :type payload: Optional[Dict[str, Any]]
        :param event_id: A unique identifier for the event. Defaults to an empty string.
        :type event_id: str
        :param kwargs: Additional keyword arguments to be passed.
        :type kwargs: Any
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
    Handles the management and processing of events with detailed functionality for
    event handling, trace management, and observability.

    The primary purpose of this class is to manage event lifecycle methods, such as
    starting and ending events, while maintaining in-flight events data. Additionally,
    the class supports trace management, enabling tracking of operations sequences.

    :ivar event_starts_to_ignore: A list of event start identifiers that should be ignored.
    :type event_starts_to_ignore: list
    :ivar event_ends_to_ignore: A list of event end identifiers that should be ignored.
    :type event_ends_to_ignore: list
    :ivar in_flight_events: A dictionary to maintain events that are currently in flight.
    :type in_flight_events: dict
    """

    def __init__(self, event_starts_to_ignore=[], event_ends_to_ignore=[]):
        """
        Initializes the object with specified events to ignore and initializes the
        in-flight events dictionary.

        :param event_starts_to_ignore: List of event start names to ignore. Defaults to an empty list.
        :type event_starts_to_ignore: list[str]
        :param event_ends_to_ignore: List of event end names to ignore. Defaults to an empty list.
        :type event_ends_to_ignore: list[str]
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
        Handles the start of an event by processing the event type and payload data.
        It stores relevant event information in-flight events for further usage.
        This function is used to manage and monitor events of specific types.

        :param event_type: The type of event being started.
        :type event_type: CBEventType
        :param payload: Optional dictionary containing event-related data.
            Defaults to None if no payload data is provided.
        :type payload: Optional[Dict[str, Any]]
        :param event_id: A unique string identifier for the event.
            Defaults to an empty string if not specified.
        :type event_id: str
        :param parent_id: A string identifier for the parent event,
            if applicable. Defaults to an empty string if not specified.
        :type parent_id: str
        :param kwargs: Additional keyword arguments to handle optional parameters.
        :type kwargs: Any
        :return: The ID of the event being processed.
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
        Handles the completion of an event and performs necessary operations such
        as computing event duration and dispatching it for processing. The method
        removes the event from the in-flight events tracker and posts it to the
        observability queue if applicable. Ensures specific cases for different
        event types, such as LLM and EMBEDDING events, are managed appropriately.

        :param event_type: The type of event that has ended, represented as
            a member of `CBEventType` enumeration.
        :param payload: The payload associated with the event, which may
            contain additional context such as messages or embeddings. Defaults
            to None.
        :param event_id: A unique identifier for the event being processed.
            Defaults to an empty string.
        :param kwargs: Additional keyword arguments that may be required for
            future extensibility.
        :return: This method does not return anything.
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
        Resets the event counts by clearing the internal data structure. This method
        empties the dictionary `in_flight_events` to ensure no previous state or data
        remains.

        :return: None
        """
        self.in_flight_events = {}

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """
        Starts a new trace session with an optional trace identifier.

        This method initiates a tracing session where a unique trace ID can be
        assigned. If no ID is provided, a default mechanism may generate one.
        Traces are fundamental in distributed systems for tracking request flows
        throughout various system components.

        :param trace_id: An optional unique identifier string for the trace.
            If not provided, the system may assign a default identifier.
        :type trace_id: Optional[str]

        :return: None
        """
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Ends the trace associated with a specific trace ID and updates the provided trace map.
        Typically used to finalize tracing activities by removing or finalizing trace entries.

        :param trace_id: Optional; Identifier for the trace that needs to be ended. If None,
            no specific trace will be targeted.
        :param trace_map: Optional; A dictionary mapping trace IDs to a list of associated
            trace details. Can be updated or modified as a part of ending the trace.
        :return: Nothing is returned by this function.
        """
        pass
