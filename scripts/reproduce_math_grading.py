
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.math_grader import grade_math, last_boxed_only_string, remove_boxed, normalize_final_answer, is_equiv, SYMPY_AVAILABLE

print(f"SymPy Available: {SYMPY_AVAILABLE}")

def test_case(name, prediction, reference):
    print(f"\n--- Test Case: {name} ---")
    print(f"Prediction: {prediction!r}")
    print(f"Reference: {reference!r}")

    boxed = last_boxed_only_string(prediction)
    print(f"Boxed extracted: {boxed!r}")

    if boxed:
        unboxed = remove_boxed(boxed)
        print(f"Unboxed: {unboxed!r}")
        norm_pred = normalize_final_answer(unboxed)
        print(f"Normalized Pred: {norm_pred!r}")
    else:
        print("No boxed answer found")
        norm_pred = normalize_final_answer(prediction) # fallback behavior of grader might be different, but let's see. logic in grade_math tries extract_last_number if no boxed.
        
        # Mimic grade_math fallback
        from src.evaluation.math_grader import extract_last_number
        extracted = extract_last_number(prediction)
        print(f"Fallback extracted: {extracted!r}")
        if extracted:
            norm_pred = normalize_final_answer(extracted)
        else:
            norm_pred = None

    norm_ref = normalize_final_answer(reference)
    print(f"Normalized Ref: {norm_ref!r}")

    graded = grade_math(prediction, reference)
    print(f"Grade Result: {graded}")
    return graded

# Case 1: The 'a' stripping issue
test_case("(a+5)", r"So the answer is \boxed{(a + 5)(b + 2)}\n$$", "(a+5)(b+2)")

# Case 2: The 'East' casing issue
test_case("East", r"Final Answer\n\n$$\n\boxed{East}\n$$", r"\text{east}")
