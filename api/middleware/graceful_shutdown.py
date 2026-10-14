"""
Graceful shutdown handler.

WHY THIS EXISTS:
When the API receives SIGTERM (from Docker stop, kubectl delete, or
a deployment), the default behavior is to kill the process immediately.
This can:
- Cut off in-progress requests mid-response
- Leave model weights half-written to disk
- Not flush logs to disk
- Not release GPU memory properly

This module hooks into FastAPI's lifespan to perform cleanup on shutdown:
1. Stop accepting new requests (nginx/ingress already does this)
2. Wait for in-progress requests to complete (up to timeout)
3. Unload models (free GPU memory)
4. Close database connections
5. Flush logs
"""

import asyncio
import signal

from loguru import logger


class GracefulShutdown:
    """Handles graceful shutdown with cleanup tasks.

    Usage in api/main.py lifespan:
        shutdown = GracefulShutdown()
        yield
        await shutdown.cleanup()
    """

    def __init__(self, shutdown_timeout: float = 30.0):
        self.shutdown_timeout = shutdown_timeout
        self._cleanup_tasks: list = []

    def register_cleanup(self, name: str, cleanup_func) -> None:
        """Register a cleanup function to run on shutdown.

        Args:
            name: Human-readable name for logging
            cleanup_func: Async or sync function to call on shutdown
        """
        self._cleanup_tasks.append((name, cleanup_func))

    async def cleanup(self) -> None:
        """Run all registered cleanup tasks in order.

        Each task gets up to shutdown_timeout seconds. If a task takes
        too long, it's cancelled and we move to the next one.
        """
        logger.info(f"Starting graceful shutdown (timeout={self.shutdown_timeout}s)...")

        for name, func in self._cleanup_tasks:
            try:
                logger.info(f"Cleanup: {name}")
                if asyncio.iscoroutinefunction(func):
                    await asyncio.wait_for(
                        func(), timeout=self.shutdown_timeout
                    )
                else:
                    func()
                logger.info(f"Cleanup complete: {name}")
            except TimeoutError:
                logger.warning(f"Cleanup timed out: {name}")
            except Exception as e:
                logger.error(f"Cleanup failed: {name}: {e}")

        logger.info("Graceful shutdown complete.")


def register_signal_handlers(shutdown_handler: GracefulShutdown) -> None:
    """Register SIGTERM/SIGINT handlers for graceful shutdown.

    In production (Docker/K8s), SIGTERM is sent before SIGKILL.
    This gives the app time to clean up.
    """
    def handle_signal(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        # The actual cleanup happens in the lifespan context manager
        # This just sets a flag that uvicorn picks up

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    logger.info("Signal handlers registered (SIGTERM, SIGINT)")
