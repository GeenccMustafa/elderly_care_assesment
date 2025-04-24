# css_styles.py

# CSS string for Gradio styling
css = """
/* Make markdown links look more like buttons */
.gr-markdown a {
    display: inline-block;
    padding: 2px 8px;
    border: 1px solid #0b5ed7; /* Bootstrap primary blue */
    border-radius: 0.25rem;
    color: #0b5ed7;
    text-decoration: none;
    background-color: white;
    transition: background-color 0.2s ease, color 0.2s ease;
    margin-left: 5px; /* Add some space */
}
.gr-markdown a:hover {
    background-color: #0b5ed7;
    color: white;
}
/* Style the notification list items */
.gr-markdown ul { padding-left: 20px; }
.gr-markdown li { margin-bottom: 5px; }

/* Increase font size for H1 and H2 slightly more */
.gr-markdown h1 {
    font-size: 2em;
    margin-top: 0.5em;
    margin-bottom: 0.5em;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.2em;
}
.gr-markdown h2 {
    font-size: 1.6em;
    margin-top: 1em;
    margin-bottom: 0.5em;
}

#questions_container .question-text-md p {
    font-size: 1.2em !important;
    font-weight: 600 !important;
    margin-top: 0.2em !important;
    margin-bottom: 8px !important;
    color: var(--body-text-color) !important;
    line-height: 1.4 !important;
}
#questions_container .question-text-md {
    padding: 0 !important;
    margin: 0 !important;
}
#questions_container .gr-row {
    margin-top: 0px !important;
}
/* --- NEW RULES for View Navigation Buttons --- */
.view-nav-button {
    font-size: 1.25em !important; /* Even larger font */
    font-weight: 600 !important; /* Bolder */
    padding: 12px 18px !important; /* More padding */
    margin: 0px 4px 10px 4px !important; /* Add bottom margin */
    border-radius: 8px !important; /* Slightly more rounded */
    flex-grow: 1; /* Make buttons expand equally */
}

.view-nav-row {
    border-bottom: 2px solid var(--border-color-primary);
    margin-bottom: 20px; /* Space below the buttons */
    padding-bottom: 5px; /* Space between buttons and border */
}

/* Optional: Style the active button differently if variant isn't enough */
/* (This might require more complex selectors depending on Gradio version) */
/* .view-nav-button.primary { ... } */

"""