import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import App from "./App.jsx";
import OverviewPage from "./pages/OverviewPage.jsx";
import DataPage from "./pages/DataPage.jsx";
import EmbeddingPage from "./pages/EmbeddingPage.jsx";
import "./styles.css";

// HashRouter so deep links work on plain static hosting (no server rewrites).
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <HashRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<OverviewPage />} />
          <Route path="data" element={<DataPage />} />
          <Route path="embeddings" element={<EmbeddingPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  </React.StrictMode>
);
