"""Simulador mínimo do Streamlit para rodar o app.py sem a biblioteca.

Registra o que o app desenha em `log` como tuplas (tipo, texto) e devolve valores configuráveis para os
widgets. Prova que o fluxo do app funciona com o backend real; não prova o comportamento visual do Streamlit.
"""

from __future__ import annotations


class Stop(Exception):
    """st.stop() interrompe a execução do script."""


class Rerun(Exception):
    """st.rerun() reinicia o script."""


class _Ctx:
    def __init__(self, fake, kind, text=""):
        self.fake, self.kind, self.text = fake, kind, text

    def __enter__(self):
        self.fake.log.append((self.kind, self.text))
        return self

    def __exit__(self, *exc):
        return False


class _Placeholder:
    def __init__(self, fake):
        self.fake = fake

    def markdown(self, text):
        self.fake.log.append(("placeholder", text))


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FakeStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.log: list[tuple[str, str]] = []
        self.prompt: str | None = None
        self.inputs: dict[str, str] = {}      # rótulo do text_input -> valor
        self.choices: dict[str, str] = {}     # rótulo do selectbox -> opção escolhida
        self.options: dict[str, list] = {}    # rótulo do selectbox -> opções oferecidas
        self.sidebar = _Ctx(self, "sidebar")

    # -- helpers de teste
    def texts(self, kind: str) -> list[str]:
        return [text for k, text in self.log if k == kind]

    # -- widgets
    def set_page_config(self, **_):
        self.log.append(("config", ""))

    def title(self, text):
        self.log.append(("title", text))

    def caption(self, text):
        self.log.append(("caption", text))

    def subheader(self, text):
        self.log.append(("subheader", text))

    def divider(self):
        pass

    def markdown(self, text):
        self.log.append(("markdown", text))

    def error(self, text):
        self.log.append(("error", text))

    def warning(self, text):
        self.log.append(("warning", text))

    def info(self, text):
        self.log.append(("info", text))

    def code(self, text, language=None):
        self.log.append(("code", text))

    def text_input(self, label, value="", **_):
        return self.inputs.get(label, value)

    def selectbox(self, label, options, **_):
        self.options[label] = list(options)
        return self.choices.get(label, options[0])

    def slider(self, label, min_value=0, max_value=100, value=0, **_):
        return value

    def button(self, label, **_):
        return False

    def rerun(self):
        raise Rerun

    def stop(self):
        raise Stop

    def chat_message(self, role):
        return _Ctx(self, "chat_message", role)

    def chat_input(self, placeholder=""):
        prompt, self.prompt = self.prompt, None  # entrega a pergunta uma única vez
        return prompt

    def spinner(self, text=""):
        return _Ctx(self, "spinner", text)

    def expander(self, label):
        return _Ctx(self, "expander", label)

    def empty(self):
        return _Placeholder(self)
