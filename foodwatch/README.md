# FoodWatch Kenya dashboard

    pip install -r requirements.txt
    python precompute.py     # one-off, ~1 minute: builds data/results.json from the CSV
    python app.py            # then open http://127.0.0.1:5000

Chart.js is bundled in static/, so the dashboard works offline.
Re-run precompute.py whenever you update data/wfp_food_prices_ken.csv.
