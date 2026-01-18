
import dspy
import sys
import os

# Add project root to sys.path to allow imports from experiments.*
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pydantic import BaseModel, Field
from experiments.dspy.adapters.toon import ToonAdapter

# Configure DSPy with ToonAdapter
# We'll use a dummy LM to test the prompt formatting, 
# and then manually test parsing with a mock response.
lm = dspy.LM("openai/gpt-4o-mini", api_key="dummy")
adapter = ToonAdapter()
dspy.settings.configure(lm=lm, adapter=adapter)

class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int

class ExtractPerson(dspy.Signature):
    """Extract person from text."""
    text: str = dspy.InputField()
    person: Person = dspy.OutputField()

def test_formatting():
    print("Testing Formatting...")
    extractor = dspy.Predict(ExtractPerson)
    # This won't actually call the API if we don't invoke it, 
    # but let's check the formatting logic directly via the adapter methods.
    
    signature = ExtractPerson
    inputs = {"text": "Alice is 30 years old."}
    
    prompt = adapter.format(signature, demos=[], inputs=inputs)
    print("--- Formatted Prompt ---")
    print(prompt)
    print("------------------------")
    
    # helper to get text from prompt
    if isinstance(prompt, list):
        prompt_text = "\n".join([str(msg.get("content", "")) for msg in prompt])
    else:
        prompt_text = str(prompt)

    # Check for TOON specific markers
    assert "TOON Format (NOT JSON)" in prompt_text
    assert "Output structure:" in prompt_text
    assert "person:" in prompt_text
    assert "name: string" in prompt_text
    assert "text: Alice is 30 years old." in prompt_text
    print("Formatting check passed!")

def test_parsing():
    print("\nTesting Parsing...")
    signature = ExtractPerson
    
    # Mock LLM output in TOON format
    completion = """
person:
  name: Alice
  age: 30
"""
    parsed = adapter.parse(signature, completion)
    print("--- Parsed Result ---")
    print(parsed)
    print("---------------------")
    
    assert "person" in parsed, f"Expected 'person' in parsed result, got: {parsed}"
    person = parsed["person"]
    assert isinstance(person, Person), f"Expected Person instance, got {type(person)}"
    assert person.name == "Alice", f"Expected Alice, got {person.name}"
    assert person.age == 30, f"Expected 30, got {person.age}"
    print("Parsing check passed!")

def test_tabular_parsing():
    print("\nTesting Tabular Parsing...")
    
    class PersonList(dspy.Signature):
        """Extract people."""
        text: str = dspy.InputField()
        people: list[Person] = dspy.OutputField()

    signature = PersonList
    
    # Mock LLM output in TOON tabular format
    completion = """
people[2]{name,age}:
  Alice,30
  Bob,25
"""
    parsed = adapter.parse(signature, completion)
    print("--- Parsed Result (Tabular) ---")
    print(parsed)
    print("-------------------------------")
    
    assert "people" in parsed
    people = parsed["people"]
    assert len(people) == 2
    assert people[0].name == "Alice"
    assert people[0].age == 30
    assert people[1].name == "Bob"
    assert people[1].age == 25
    print("Tabular parsing check passed!")

if __name__ == "__main__":
    try:
        test_formatting()
        test_parsing()
        test_tabular_parsing()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
