import React, { Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import App from "./App.jsx";
import OverviewPage from "./pages/OverviewPage.jsx";
import DataPage from "./pages/DataPage.jsx";
import "./styles.css";

// Lazy-load the Plotly-heavy embedding page so its big bundle only loads when
// that route is visited.
const EmbeddingPage = lazy(() => import("./pages/EmbeddingPage.jsx"));
const MicroscopePage = lazy(() => import("./pages/MicroscopePage.jsx"));

// HashRouter so deep links work on plain static hosting (no server rewrites).
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <HashRouter>
      <Suspense fallback={<div className="content muted">Loading…</div>}>
        <Routes>
          <Route path="/" element={<App />}>
            <Route index element={<OverviewPage />} />
            <Route path="data" element={<DataPage />} />
            <Route path="embeddings" element={<EmbeddingPage />} />
            <Route path="microscope" element={<MicroscopePage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </HashRouter>
  </React.StrictMode>
);
