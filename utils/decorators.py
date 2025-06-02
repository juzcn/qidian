import asyncio
import functools
import inspect
import logging
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any, Callable, TypeVar, get_type_hints, get_origin, get_args, Union, Coroutine

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

T = TypeVar('T')


def function_logging(
        enable_type_checking: bool = True,
#        log_level: int = logging.DEBUG,
        handle_errors: bool = True,
        max_arg_length: int = None,
        log_yield: bool = False
):
    """
    Decorator factory for logging function execution details.

    Args:
        enable_type_checking: Whether to enable type checking of arguments
        log_level: Logging level to use
        handle_errors: Whether to handle and log errors or let them propagate
        max_arg_length: int = None, log length positive first N, negative last N
        log_yield: bool = True， log each yield
    """

    def _format_value(value: Any) -> str:
        """Format values for logging with truncation if needed"""
        str_val = str(value)
        if max_arg_length is None or len(str_val) <= abs(max_arg_length):
            return f"{str_val} ({type(value).__name__})"
        if max_arg_length >= 0:
            return f"{str_val[:max_arg_length]}... ({type(value).__name__})"
        return f"...{str_val[max_arg_length:]} ({type(value).__name__})"

    def _check_type(value: Any, expected_type: Any) -> bool:
        """Helper function to check if value matches expected type, including generic types."""
        if expected_type is Any or inspect.isasyncgen(expected_type):
            return True
        # Handle Optional[T] and Union[T, None]
        if get_origin(expected_type) is AsyncGenerator:
            # pass
            return True
        if get_origin(expected_type) is Union:
            type_args = get_args(expected_type)
            if type(None) in type_args:  # Optional[T] case
                if value is None:
                    return True
                # Check other types in the Union
                return any(_check_type(value, t) for t in type_args if t is not type(None))

        # Handle other Union types
        if get_origin(expected_type) is Union:
            return any(_check_type(value, t) for t in get_args(expected_type))

        # Handle generic types like List[int]
        origin_type = get_origin(expected_type)
        if origin_type is not None:
            if not isinstance(value, origin_type):
                return False
            type_args = get_args(expected_type)
            if not type_args:  # No generic parameters specified
                return True

            # Check each item in the container
            if origin_type in (list, tuple, set):
                return all(_check_type(item, type_args[0]) for item in value)
            elif origin_type is dict:
                return all(
                    _check_type(k, type_args[0]) and _check_type(v, type_args[1])
                    for k, v in value.items()
                )
            return True

        # Handle simple types
        return isinstance(value, expected_type)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Get function signature for parameter names
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        return_type = type_hints.get('return', Any)
        func_name = func.__name__
        # Extract AsyncGenerator type parameters if present
        yield_type = Any
        send_type = Any
        return_gen_type = Any
        if get_origin(return_type) is AsyncGenerator or Generator:
            type_args = get_args(return_type)
            if len(type_args) >= 2:
                yield_type, send_type = type_args[:2]
            if len(type_args) >= 3:
                return_gen_type = type_args[2]

        def _validate_input_types(bound_args):
            """Validate input argument types"""
            for param_name, param_value in bound_args.arguments.items():
                if param_name in type_hints:
                    expected_type = type_hints[param_name]
                    if not _check_type(param_value, expected_type):
                        error_msg = (
                            f"TypeError: Argument '{param_name}' should be of type {expected_type}, "
                            f"got {type(param_value)} instead"
                        )
                        raise TypeError(error_msg)

        def _validate_return_type(value, expected_type, is_yield=False):
            """Validate return or yield value types"""
            if not _check_type(value, expected_type):
                value_type = type(value)
                if is_yield:
                    error_msg = (
                        f"TypeError: Generator '{func_name}' should yield type {expected_type}, "
                        f"got {value_type} instead"
                    )
                else:
                    error_msg = (
                        f"TypeError: Function '{func_name}' should return type {expected_type}, "
                        f"got {value_type} instead"
                    )
                raise TypeError(error_msg)

        def _log_input(bound_args):
            for param, value in bound_args.arguments.items():
                param_info = sig.parameters[param]
                #                    truncated_val = str(value)[:max_arg_length] + ('...' if len(str(value)) > max_arg_length else '')
                logger.debug(
                           f"    • {param}: {_format_value(value)}"
                           f"{' [default]' if param in bound_args.kwargs and param_info.default == value else ''}"
                           )

        # Determine the type of function we're decorating
        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> AsyncGenerator:
                start_time = time.perf_counter()
                items_yielded = 0
                # Log inputs
                logger.debug(f"Calling async generator {func_name} with arguments:")
                # Bind arguments to parameter names
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Log all arguments with their names
                _log_input(bound_args)

                if enable_type_checking:
                    try:
                        _validate_input_types(bound_args)
                    except TypeError as e:
                        logger.error(f"Type validation failed for {func_name}: {str(e)}")
                        if not handle_errors:
                            raise

                gen = func(*args, **kwargs)

                try:
                    async for item in gen:
                        if enable_type_checking and yield_type is not Any:
                            _validate_return_type(item, yield_type, is_yield=True)
                        items_yielded += 1
                        if log_yield:
                            logger.debug(f"    • {func_name} yields #{items_yielded}: {_format_value(item)}")
                        yield item

                except Exception as e:
                    logger.error(f"Async generator {func_name} raised {type(e).__name__}: {str(e)}")
                    if not handle_errors:
                        raise
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    logger.debug(f"    • {func_name} Items yielded: {items_yielded}")
                    logger.debug(f"Async generator {func_name} completed in {duration:.4f} seconds")

            return async_gen_wrapper

        elif inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                result = None

                # Log inputs
                logger.debug(f"Calling async function {func_name} with args:")
                # Bind arguments to parameter names
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Log all arguments with their names
                _log_input(bound_args)

                if enable_type_checking:
                    try:
                        _validate_input_types(bound_args)
                    except TypeError as e:
                        logger.error(f"Type validation failed for {func_name}: {str(e)}")
                        if not handle_errors:
                            raise

                try:
                    result = await func(*args, **kwargs)
                    # Type checking for return value
                    if enable_type_checking and 'return' in type_hints:
                        _validate_return_type(result, type_hints['return'])

                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    logger.debug(f"Async function {func_name} returned in {duration:.4f} seconds. Result: {result}")
                    return result
                except Exception as e:
                    logger.error(f"Async function {func_name} failed with {type(e).__name__}: {str(e)}")
                    if not handle_errors:
                        raise
                return result

            return async_wrapper

        elif inspect.isgeneratorfunction(func):
            @functools.wraps(func)
            #            def gen_wrapper(*args: Any, **kwargs: Any) -> Generator:
            def gen_wrapper(*args: Any, **kwargs: Any) -> Generator:
                start_time = time.perf_counter()
                items_yielded = 0
                # Log inputs
                logger.debug(f"Calling generator {func_name} with argsuments:")
                # Bind arguments to parameter names
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Log all arguments with their names
                _log_input(bound_args)

                if enable_type_checking:
                    try:
                        _validate_input_types(bound_args)
                    except TypeError as e:
                        logger.error(f"Type validation failed for {func_name}: {str(e)}")
                        if not handle_errors:
                            raise

                gen = func(*args, **kwargs)

                try:
                    for item in gen:
                        if enable_type_checking and yield_type is not Any:
                            _validate_return_type(item, yield_type, is_yield=True)

                        items_yielded += 1
                        if log_yield:
                            logger.debug(f"    • {func_name} yields #{items_yielded}: {_format_value(item)}")
                        yield item

                except Exception as e:
                    logger.error(f"Generator {func_name} raised {type(e).__name__}: {str(e)}")
                    if not handle_errors:
                        raise
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    logger.debug(f"    • {func_name} Items yielded: {items_yielded}")
                    logger.debug(f"Generator {func_name} completed in {duration:.4f} seconds")

            return gen_wrapper

        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                result = None

                # Log inputs
                logger.debug(f"Calling function {func_name} with args:")
                # Bind arguments to parameter names
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Log all arguments with their names
                _log_input(bound_args)

                if enable_type_checking:
                    try:
                        _validate_input_types(bound_args)
                    except TypeError as e:
                        logger.error(f"Type validation failed for {func_name}: {str(e)}")
                        if not handle_errors:
                            raise

                try:
                    result = func(*args, **kwargs)
                    # Type checking for return value
                    if enable_type_checking and 'return' in type_hints:
                        _validate_return_type(result, type_hints['return'])

                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    logger.debug(f"Function {func_name} returned in {duration:.4f} seconds. Result: {result}")
                    return result
                except Exception as e:
                    logger.error(f"Function {func_name} failed with {type(e).__name__}: {str(e)}")
                    if not handle_errors:
                        raise
                return result

            return sync_wrapper

    return decorator


def sync_async_func(async_func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """Decorator to convert an async function into a synchronous function using asyncio.run().

    Preserves type hints and handles cleanup automatically.
    """
    if not inspect.iscoroutinefunction(async_func):
        raise TypeError(f"Expected coroutine function, got {type(async_func).__name__}")

    @functools.wraps(async_func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> T:
        # Create the coroutine object
        coro = async_func(*args, **kwargs)

        # Run it using asyncio.run() which handles loop creation/cleanup
        return asyncio.run(coro)

    return sync_wrapper


def sync_async_gen(func: Any) -> Any:
    """Decorator to convert an async generator function into a regular generator function.

    Preserves the original function's type hints and docstring.
    Handles both the async generator function and its resulting async generator.
    """

    # Check if the input is actually an async generator function
    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Generator[T, None, None]:
        # Create the async generator
        async_gen = func(*args, **kwargs)

        # Run the async generator in an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                try:
                    # Get the next value from the async generator
                    value = loop.run_until_complete(async_gen.__anext__())
                    yield value
                except StopAsyncIteration:
                    break
        finally:
            if loop is not None:
                try:
                    # Run any pending tasks
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending))
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
    return sync_wrapper
