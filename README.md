<h1 align="center">📈 Bitcoin Forecast Logger (ML + Web App)</h1>

<p align="center">
  A Bitcoin forecasting platform using Machine Learning models (Random Forest, LSTM, ARIMA) 
  with real-time inference, interactive UI, and deployment via Docker + Render.
</p>

<p align="center">
  <a href="https://blockchain-forecast-logger-kishen.onrender.com/" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Launch-Web%20App-blue?style=for-the-badge" />
  </a>
</p>

<hr>

<h2>📌 Overview</h2>

<p>
This project predicts the <b>next-day Bitcoin closing price</b> using a combination of Machine Learning and Deep Learning models.
It is designed as a <b>real-world, deployable application</b> with fast inference using pretrained models and optional retraining.
</p>

<ul>
  <li>📊 Forecast next-day BTC price</li>
  <li>📉 Predict market direction (Up / Down)</li>
  <li>📦 Provide prediction intervals & confidence</li>
  <li>⚡ Fast inference using pretrained models</li>
  <li>🔄 Optional model retraining with live data</li>
</ul>

<hr>

<h2>🧠 Models Used</h2>

<ul>
  <li><b>Random Forest</b> – captures nonlinear relationships and engineered features</li>
  <li><b>LSTM (Deep Learning)</b> – sequence-based time series modeling</li>
  <li><b>ARIMA</b> – statistical baseline for time series forecasting</li>
</ul>

<p>
The system intelligently handles model instability (e.g. LSTM scaling issues) using fallback logic to ensure robust predictions.
</p>

<hr>

<h2>⚙️ Key Features</h2>

<ul>
  <li>🔹 Live BTC data integration (via Yahoo Finance)</li>
  <li>🔹 Pretrained model loading for fast predictions</li>
  <li>🔹 User-triggered retraining pipeline</li>
  <li>🔹 Prediction interval visualization</li>
  <li>🔹 Interactive Streamlit dashboard</li>
  <li>🔹 Dockerized deployment</li>
</ul>

<hr>

<h2>📊 Example Output</h2>

<ul>
  <li>Latest BTC Price</li>
  <li>Predicted Next-Day Price</li>
  <li>Direction (Up / Down)</li>
  <li>Confidence (%)</li>
  <li>Prediction Interval</li>
</ul>

<hr>

<h2>🚀 Tech Stack</h2>

<ul>
  <li><b>Python</b> – Core language</li>
  <li><b>pandas / NumPy</b> – Data processing</li>
  <li><b>scikit-learn</b> – Random Forest</li>
  <li><b>TensorFlow / Keras</b> – LSTM model</li>
  <li><b>statsmodels</b> – ARIMA</li>
  <li><b>Streamlit</b> – Web app UI</li>
  <li><b>Docker</b> – Containerization</li>
  <li><b>Render</b> – Deployment</li>
</ul>

<hr>

<h2>🧩 Project Structure</h2>

<pre>
.
├── bitcoin_forecast_models.py   # Model logic (RF, LSTM, ARIMA)
├── model_runner.py             # Prediction + pretrained model loader
├── streamlit_app.py            # Web app UI
├── train_models.py             # Offline training script
├── models/                     # Saved models (pkl / keras)
├── requirements.txt
├── Dockerfile
</pre>

<hr>

<h2>🛠️ How It Works</h2>

<ol>
  <li>Models are trained offline and saved</li>
  <li>App loads pretrained models instantly</li>
  <li>User selects model and generates prediction</li>
  <li>Optional: user retrains models using live data</li>
</ol>

<hr>

