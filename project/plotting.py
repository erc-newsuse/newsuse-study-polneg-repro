from collections.abc import Iterator

import arviz as az

__all__ = ("ArvizLabeller",)


class ArvizLabeller(az.labels.BaseLabeller):
    def make_label_vert(self, var_name, *_) -> str:
        kind, var_name = var_name.split("_", 1)
        term = " * ".join(self._iter_parts(var_name))
        return f"[{kind}] {term}"

    def _iter_parts(self, var_name: str) -> Iterator[str]:
        for pat in ("__", ":"):
            if pat in var_name:
                for part in var_name.split(pat):
                    yield from self._iter_parts(part)
                return
        else:
            yield self._process_var_name(var_name)

    def _process_var_name(self, var_name: str) -> str:
        if var_name.startswith(prefix := "country"):
            return var_name.removeprefix(prefix).upper().strip()
        if var_name.startswith(prefix := "political"):
            return var_name.removeprefix(prefix).upper().strip()
        if var_name.startswith(prefix := "Intercept"):
            return "Threshold" + var_name.removeprefix(prefix)
        return var_name
