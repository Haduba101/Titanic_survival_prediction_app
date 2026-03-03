# Prompt to Generate index.html

Create an HTML file for a Flask app called "Titanic Survival Predictor". The page should have:

## Styling

- Use embedded CSS (no external files)
- Blue gradient background (`#1e3c72` to `#2a5298` at 135deg)
- Centered white container card with rounded corners (15px), padding (30px), and box shadow
- Max width of 500px
- Arial font family
- Form inputs with 2px solid `#ddd` borders that turn blue (`#2a5298`) on focus
- Submit button with the same gradient as the background, white text, hover effect that scales to 1.02

## Content

- Title "Titanic Survival Predictor" as h1 (centered, color `#1e3c72`)
- Subtitle paragraph: "Enter passenger details to predict survival"
- Yellow info box with left border (4px solid `#ffc107`) containing a tip about Random Forest model

## Form (POST to `/predict`)

1. **Ticket Class** (select, name="pclass"): Options for 1st/2nd/3rd class with values 1/2/3, 3rd class selected by default
2. **Gender** (select, name="sex"): Male=0, Female=1
3. **Age** (number input, name="age"): min 0, max 100, default 25
4. **Siblings/Spouses Aboard** (number input, name="sibsp"): min 0, max 10, default 0
5. **Parents/Children Aboard** (number input, name="parch"): min 0, max 10, default 0
6. **Ticket Fare** (number input, name="fare"): min 0, max 600, step 0.01, default 32.00

Each label should have a hint in parentheses explaining the field (styled smaller and gray).

Submit button text: "🔮 Predict Survival"

## Results Section

- Use Jinja2 templating
- Conditionally display a result div if `prediction_text` exists
- Show `{{ prediction_text }}` as h2
- If `confidence_text` exists, show it as a paragraph below
