from __future__ import annotations

import multiprocessing
import time
from typing import Protocol, runtime_checkable

from ageval.core import Agent, Prediction, Task


@runtime_checkable
class Sandbox(Protocol):
    """Process-isolation boundary for agent execution.

    Implementations run an agent's ``predict()`` in an isolated child
    process so that crashes, infinite loops, or resource exhaustion
    in the agent do not bring down the host process.
    """

    def run(self, agent: Agent, task: Task, timeout: float) -> Prediction: ...


def _worker(agent: Agent, task: Task, queue: multiprocessing.Queue) -> None:
    """Invoke ``agent.predict(task)`` and post the result onto *queue*."""
    try:
        prediction = agent.predict(task)
        queue.put(prediction)
    except Exception as e:
        queue.put(Prediction(error=str(e)))


class SubprocessSandbox:
    """Runs ``agent.predict()`` in a child process with a hard timeout.

    **The agent and task must be picklable** — arbitrary closures,
    lambdas, and bound methods that capture non-picklable state are
    **unsupported** and will surface as an error-carrying ``Prediction``.

    On timeout or crash the child is terminated and a ``Prediction`` with
    ``error`` set is returned. This class never raises from ``run()``.
    """

    def __init__(self, context: str | None = None) -> None:
        """Initialize the sandbox.

        Args:
            context: multiprocessing start method (``"forkserver"``,
                ``"spawn"``, etc.). ``None`` lets the standard library
                pick the default context for the platform.
        """
        self._context = context

    def run(self, agent: Agent, task: Task, timeout: float) -> Prediction:
        """Run ``agent.predict(task)`` in a child process.

        Args:
            agent: the agent to invoke. Must be picklable.
            task: the task to pass to the agent.
            timeout: maximum seconds to wait for a result.

        Returns:
            The ``Prediction`` produced by the agent, or a
            ``Prediction(error=...)`` if the agent timed out, crashed,
            or could not be dispatched.
        """
        try:
            ctx = self._get_context()
            queue: multiprocessing.Queue = ctx.Queue()
            process = ctx.Process(
                target=_worker,
                args=(agent, task, queue),
                daemon=True,
            )
        except Exception as e:
            return Prediction(error=f"sandbox setup failed: {e}")

        start = time.perf_counter()

        try:
            process.start()
            try:
                return queue.get(timeout=timeout)
            except Exception:
                elapsed = time.perf_counter() - start
                return Prediction(error=f"timeout after {elapsed:.1f}s")
        except Exception as e:
            return Prediction(error=f"sandbox error: {e}")
        finally:
            self._cleanup(process)

    def _get_context(self) -> multiprocessing.context.BaseContext:
        if self._context is None:
            return multiprocessing.get_context()
        return multiprocessing.get_context(self._context)

    @staticmethod
    def _cleanup(process: multiprocessing.Process) -> None:
        """Terminate and join the child process.

        First calls ``terminate()`` (SIGTERM), waits briefly, then
        ``kill()`` (SIGKILL) if still alive.
        """
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            else:
                process.join(timeout=0.1)
        except Exception:
            pass
        finally:
            if process.is_alive():
                try:
                    process.kill()
                    process.join(timeout=1.0)
                except Exception:
                    pass
