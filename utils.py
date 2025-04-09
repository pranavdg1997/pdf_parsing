from typing import List, Dict

def assign_sections(
    pages: List[Dict],          # [{'content': "...", 'is_scanned': bool, ...}, …]
    sections: List[str],        # ['introduction', 'methodology', 'results', …]
    fallback: str = "unknown"   # what to call pages that precede the 1st section
) -> List[Dict]:
    """
    Adds a key ``section`` to every page‑dict and returns the modified list.
    Pages keep whatever keys they already had.

    Rules
    -----
    1. The first time a section title (case‑insensitive) appears in a page’s
       text, that page starts the section.
    2. Pages belong to that section until a *later* section title is found.
       Any sections whose titles never appear are skipped automatically.
    3. If ``is_scanned`` is *True* for any page, that page **and every page
       after it** are labeled ``"case_documents"``.
    """
    # normalise section titles once to speed things up
    section_lc = [s.lower() for s in sections]

    current_section     = fallback         # what we label the page with
    next_section_index  = 0                # index in ``sections`` we’re still hunting for
    locked_case_docs    = False            # becomes True after the first scanned page

    for page in pages:
        # ------------- 1) scanned pages override everything -------------
        if page.get("is_scanned", False):
            current_section  = "case_documents"
            locked_case_docs = True

        # ------------- 2) otherwise, try to detect (next) section -------
        elif not locked_case_docs:
            text = str(page.get("content", "")).lower()

            # Look for the first *remaining* section title that appears in this page
            for idx in range(next_section_index, len(section_lc)):
                if section_lc[idx] in text:
                    # Found the section that starts here
                    current_section    = sections[idx]
                    next_section_index = idx + 1       # only look for *later* sections now
                    break
            # If no section title is present we simply keep the previous one

        # ------------- 3) tag the page ----------------------------------
        page["section"] = current_section

    return pages
