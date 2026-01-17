"""
MATH evaluation grader using proper mathematical equivalence checking.

Based on lm-evaluation-harness minerva_math implementation:
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py

Uses SymPy for algebraic equivalence checking.
"""

import re
import logging
from typing import Optional

logging.basicConfig(level=logging.WARNING)
eval_logger = logging.getLogger(__name__)

# Try to import sympy, but make it optional
try:
    import sympy
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    eval_logger.warning("SymPy not available. Using fallback string comparison for MATH grading.")


def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extract the last boxed answer from a string.
    Handles both \\boxed{} and \\fbox{} formats.
    """
    if not string:
        return None

    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    else:
        return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Remove the \\boxed{} or \\boxed wrapper from a string."""
    if not s:
        return ""

    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]
        return s

    left = "\\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]

    return s


# Substitutions for normalization
SUBSTITUTIONS = [
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

# Expressions to remove
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\\circ",
    r"^{\\circ}",
    r"\\;",
    r",\\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Based on Appendix D of Lewkowycz et al. (2022) - Minerva paper.
    """
    if not final_answer:
        return ""

    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    
    # Use regex with word boundaries to avoid corrupting latex commands
    # e.g. "ft" in "\left" should not be removed
    for expr in REMOVED_EXPRESSIONS:
        # Escape expr just in case it contains special regex chars, though most don't
        # We use \b to match word boundaries
        pattern = r"\b" + re.escape(expr) + r"\b"
        final_answer = re.sub(pattern, "", final_answer)

    # Extract answer that is in LaTeX math, is bold, is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\\$)(.*?)(\\$)(.*)", r"$\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    # \dfrac -> \frac
    final_answer = final_answer.replace(r"\dfrac", r"\frac")
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def is_equiv(x1: str, x2: str, timeout_seconds: int = 5) -> bool:
    """
    Check if two normalized LaTeX strings are mathematically equivalent.
    Uses SymPy to check if their difference simplifies to 0.
    """
    if not SYMPY_AVAILABLE:
        # Fallback to string comparison
        return x1.strip() == x2.strip()

    try:
        # Try to parse both expressions
        try:
            parsed_x1 = parse_latex(x1)
            parsed_x2 = parse_latex(x2)
        except (
            sympy.parsing.latex.errors.LaTeXParsingError,
            sympy.SympifyError,
            TypeError,
            AttributeError,
        ):
            eval_logger.debug(f"Couldn't parse one of {x1} or {x2}")
            # Fallback to normalized string comparison
            return x1.strip() == x2.strip()

        # Try to compute difference
        try:
            diff = parsed_x1 - parsed_x2
        except TypeError:
            eval_logger.debug(f"Couldn't subtract {x1} and {x2}")
            return False

        # Try to simplify and check if it's zero
        try:
            simplified = sympy.simplify(diff)
            return simplified == 0
        except (ValueError, TypeError) as e:
            eval_logger.debug(f"Had trouble simplifying when comparing {x1} and {x2}: {e}")
            # Fallback to string comparison
            return x1.strip() == x2.strip()

    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        # Robust fallback: if anything goes wrong (e.g. missing antlr4, recursion depth), 
        # default to string comparison
        return x1.strip() == x2.strip()


def extract_last_number(text: str) -> Optional[str]:
    """Fallback: extract last number from text if no boxed answer found."""
    if not text:
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if matches:
        return matches[-1]
    return None


def grade_math(prediction: str, reference: str) -> bool:
    """
    Grade a MATH problem prediction against the reference answer.

    Args:
        prediction: Model-generated solution (should contain \\boxed{answer})
        reference: Ground truth answer from MATH-500 dataset

    Returns:
        True if the answers are mathematically equivalent, False otherwise.
    """
    if not prediction:
        return False

    # Extract the boxed answer from prediction
    boxed_pred = last_boxed_only_string(prediction)

    if boxed_pred:
        # Remove the boxed wrapper
        pred_answer = remove_boxed(boxed_pred)
    else:
        # Fallback: try to extract last number
        pred_answer = extract_last_number(prediction)
        if pred_answer is None:
            return False

    # Normalize both answers
    norm_pred = normalize_final_answer(pred_answer)
    norm_ref = normalize_final_answer(reference)

    # Check equivalence
    if SYMPY_AVAILABLE:
        if is_equiv(norm_pred, norm_ref):
            return True
    
    # Simple string comparison fallback
    if norm_pred.strip() == norm_ref.strip():
        return True
        
    # Case-insensitive fallback for text answers
    # We only applying this if length > 1 to avoid conflating variables like 'A' and 'a'.
    # This handles answers like "East" vs "east" or "True" vs "true".
    if len(norm_ref.strip()) > 1 and norm_pred.strip().lower() == norm_ref.strip().lower():
        return True
        
    return False
