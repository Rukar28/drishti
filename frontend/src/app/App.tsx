import { useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { startRealtime } from "./realtime";
import { DeviceStatus } from "./components";
import Entry from "./pages/Entry";
import Layout from "./pages/Layout";
import Home from "./pages/Home";
import Assist from "./pages/Assist";
import Navigation from "./pages/Navigation";
import Safety from "./pages/Safety";
import History from "./pages/History";
import Overview from "./pages/Overview";
import Settings from "./pages/Settings";
export default function App() {
  useEffect(() => {
    document.documentElement.classList.toggle(
      "large-text",
      localStorage.getItem("vm-large") === "true",
    );
    return startRealtime();
  }, []);
  return (
    <Routes>
      <Route path="/" element={<Entry />} />
      <Route path="/admin/login" element={<Entry />} />
      <Route element={<Layout />}>
        <Route path="/home" element={<Home />} />
        <Route path="/assist" element={<Assist />} />
        <Route path="/navigate" element={<Navigation />} />
        <Route path="/safety" element={<Safety />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/admin" element={<Overview />} />
        <Route path="/admin/location" element={<Navigation caregiver />} />
        <Route path="/admin/safety" element={<Safety />} />
        <Route path="/admin/history" element={<History />} />
        <Route
          path="/admin/device"
          element={
            <>
              <h1>Device health</h1>
              <DeviceStatus />
            </>
          }
        />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
