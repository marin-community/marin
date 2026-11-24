import { createRoot } from "react-dom/client";
import "@marina/tokens.css";
import "./style.css";
import { App } from "./App";

createRoot(document.getElementById("app")!).render(<App />);
