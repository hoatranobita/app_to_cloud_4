    gcloud builds submit --tag gcr.io/plotly-dash-rohit/plotly-dash-rohit --project=dash-plotly-testing
    gcloud run deploy --image gcr.io/plotly-dash-rohit/plotly-dash-rohit --platform managed --project=plotly-dash-rohitg --allow-unauthenticated
