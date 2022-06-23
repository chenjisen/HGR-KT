import sys
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, BooleanOptionalAction, Namespace

from pprint import pformat
from typing import Any, Callable, Optional, Type


class Args(Namespace, metaclass=ABCMeta):
    @abstractmethod
    def process_args(self) -> None:
        pass

    def set_args(self, args: dict[str, Any]) -> None:
        for k, v in args.items():
            setattr(self, k, v)

    def as_dict(self) -> dict[str, Any]:
        return as_dict(self)

    def __str__(self) -> str:
        return pformat(self.as_dict(), sort_dicts=False)


class MyParser(ArgumentParser):

    def __init__(self, args_class: Type[Args]) -> None:
        super().__init__()
        self.args_class = args_class
        self.add_argument_from_class(args_class)

    def add_argument_from_class(self, obj: Any) -> None:
        parser = self.add_argument_group(obj.__class__.__name__)
        for k, v in as_dict(obj).items():
            parser.add_argument(
                '--' + k, type=type(v), default=v,
                action=BooleanOptionalAction if isinstance(v, bool) else None)

    def parse_args(self, kwargs: Optional[dict[str, Any]] = ...) -> Any:
        if kwargs.get('known'):
            parsed_args, _ = super().parse_known_args(get_args(kwargs))
        else:
            parsed_args = super().parse_args(get_args(kwargs))
        args = self.args_class()
        args.set_args(vars(parsed_args))
        args.process_args()
        return args

    def parse_model_name(self, kwargs: Optional[dict[str, Any]] = ...) -> Any:
        temp_args, _ = super().parse_known_args(get_args(kwargs))
        return temp_args.model


def get_args(kwargs: Optional[dict[str, Any]] = ...) -> list[str]:
    args_from_kwargs = []
    for k, v in kwargs.items():
        if isinstance(v, bool):
            args_from_kwargs.append(f"--{'' if v else 'no-'}{k}")
        else:
            args_from_kwargs.append(f'--{k}={v}')
    args = args_from_kwargs + sys.argv[1:]
    return args


def as_dict(obj) -> dict[str, Any]:
    d = {var: getattr(obj, var) for var, val in vars(obj).items()
         if not (var.startswith('_')
                 or isinstance(val, Callable)
                 or isinstance(val, property)
                 or isinstance(val, staticmethod))}
    return d
