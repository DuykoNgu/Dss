import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./common.css";
import "./features/market/market.css";
import "./features/market/analysis.css";
import "./features/discovery/discovery.css";
import "./features/research/research.css";
import "./layout.css";
import "./features/assistant/assistant.css";
import "./responsive.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
