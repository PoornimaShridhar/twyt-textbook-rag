from llm_wrapper import invoke_llm


class LLMInvoke:
    def invoke(self, p):
        return f"invoked:{p}"


class LLMPredict:
    def predict(self, p):
        return f"pred:{p}"


def test_invoke_prefers_invoke_method():
    llm = LLMInvoke()
    res = invoke_llm(llm, "hello")
    assert res == "invoked:hello"


def test_invoke_falls_back_to_predict():
    llm = LLMPredict()
    res = invoke_llm(llm, "world")
    assert res == "pred:world"
