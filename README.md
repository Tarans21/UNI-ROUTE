# UNI-ROUTE
UniRoute AI  is an intelligent campus navigation system that uses NLP, deep learning, and Dijkstra’s shortest path to guide users across multi-floor buildings. With real-time routing, room search, and 3D floor visualization, it simplifies indoor navigation and enhances accessibility on university campuses.

UniRoute – Intelligent Campus Navigation System 🚀🏫

UniRoute is an advanced AI-powered campus navigation system designed to help students, faculty, and visitors quickly locate rooms, labs, offices, and facilities inside multi-floor academic buildings.

It combines Graph-based pathfinding, Machine Learning, 3D floor visualization, and a modern React Frontend + Python Backend to deliver a smart, seamless, and interactive indoor navigation experience.

🌟 Key Features
🔍 Intelligent Search

Search rooms using natural language (e.g., “Take me to AC-403”).

Fuzzy matching for spelling mistakes.

Semantic understanding using NLP.

🧭 AI-Powered Navigation

LSTM-based next-step prediction.

Weighted Dijkstra pathfinding (with floor-change cost).

Multi-floor routing with auto-linked stairs/lifts.

Real-time generation of indoor navigation routes.

🏢 3D Campus Visualization

Interactive 3D building model.

Floor-wise view (GF, 1F, 2F, 3F, 4F).

Clear, minimal edge lines for less clutter.

Blue line highlighting for computed path.

📍 Indoor Positioning Support

QR code scan for instant location.

Signboard recognition (future expansion).

Live map panning and zooming.

🎨 Modern UI

Built using Streamlit / React (depending on build).

Clean vertical layout.

Animated transitions.

Fully responsive design.

📁 Offline Data Storage

Loads floor graphs from:

.csv files

.json cache

.pkl saved graph

Compatible with any building if CSV coordinates are provided.

🔧 Tech Stack
Frontend

React.js

Next.js

TailwindCSS

Streamlit (alternative GUI build)

Backend

Python FastAPI / Flask

TensorFlow / Keras

NetworkX

NumPy, Pandas

Plotly for 3D

Machine Learning

LSTM Next-Step Predictor

Weighted Dijkstra Routing

Graph-based heuristics

Node normalization, fuzzy logic

Data Formats

CSV (floor data)

JSON (graph cache, id maps)

Pickle (.pkl graph)

Numpy (.npy distance matrix)

🗂️ Project Structure
UniRoute/
│── backend/
│   ├── app.py
│   ├── models/
│   │   ├── lstm_nextstep.h5
│   │   ├── name2id.json
│   │   └── dijkstra_distance_matrix.npy
│   ├── graph/
│   │   ├── uniroute_graph.pkl
│   │   ├── graph_cache.json
│   │   └── floor_csvs/
│   │       ├── groundfloor.csv
│   │       ├── Floor1.csv
│   │       ├── Floor2.csv
│   │       ├── Floor3.csv
│   │       └── Floor4.csv
│
│── frontend/
│   ├── pages/
│   ├── components/
│   ├── styles/
│   └── next.config.js
│
│── README.md
└── requirements.txt

🧪 Features in Development

AR-based visual indoor navigation

Live crowd estimation

Voice-guided path instructions

Integration with campus management systems

🎯 Why UniRoute?

UniRoute solves real-world student problems:

No more confusion finding rooms

No dependency on manual maps

Accessibility for new students and visitors

It is AI-driven, scalable, and can be deployed in any educational institution with minor CSV changes.

📝 Contributing

Pull requests are welcome!
If you want to contribute:

Fork this repo

Create your feature branch

Submit PR

🛡️ License

This project is licensed under the MIT License.

💬 Support

For queries, improvements, or suggestions, feel free to raise an issue!
