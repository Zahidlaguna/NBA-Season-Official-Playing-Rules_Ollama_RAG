from querying import rag_query
from langchain_community.llms.ollama import Ollama

EVALUATION_PROMPT = '''
Expected: {expected}
Actual: {actual}
---
{answer with 'true' or 'false'} if match is correct
''' 

def test_nba_rules():
    assert query_and_validate(question='what is record keeping', expected='Record Keeping', document='2023-24-NBA-Season-Official-Playing-Rules.pdf')
    assert query_and_validate(question='what is the purpose of the rules', expected='Purpose of the Rules', document='2023-24-NBA-Season-Official-Playing-Rules.pdf')
    
def query_and_validate(question: str, expected: str, document: str):
    response_text = rag_query(question)
    prompt = EVALUATION_PROMPT.format(
        expected_response=expected, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )