import { NavLink, Outlet } from "react-router-dom";

// One nav entry per pipeline stage. New stages (transformer microscope, etc.)
// get added here as they are built.
const STAGES = [
  { to: "/", label: "Overview", end: true, step: "" },
  { to: "/data", label: "The Language", step: "Step 2" },
  { to: "/embeddings", label: "Embeddings", step: "Step 3" },
];

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="brand">
          Understanding<span>LM</span>
        </h1>
        <p className="tagline">A transformer, small enough to read by hand.</p>
        <nav>
          {STAGES.map((s) => (
            <NavLink
              key={s.to}
              to={s.to}
              end={s.end}
              className={({ isActive }) =>
                "nav-item" + (isActive ? " active" : "")
              }
            >
              {s.step && <span className="step-tag">{s.step}</span>}
              <span>{s.label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
