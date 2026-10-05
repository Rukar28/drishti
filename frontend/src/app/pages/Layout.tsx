import { Link, NavLink, Navigate, Outlet, useLocation } from "react-router-dom";
import { Eye, OctagonX, Settings as SettingsIcon } from "lucide-react";
import { api } from "../api";
import { useApp } from "../store";
import { Action, Badge } from "../components";
export default function Layout() {
  const path = useLocation().pathname,
    admin = path.startsWith("/admin");
  const online = useApp((s) => s.online),
    connection = useApp((s) => s.connection),
    error = useApp((s) => s.error),
    updated = useApp((s) => s.updated);
  const role = sessionStorage.getItem("vm-role");
  if (!role || (admin && role !== "caregiver"))
    return <Navigate to="/" replace />;
  const links = admin
    ? [
        ["/admin", "Overview"],
        ["/admin/location", "Location"],
        ["/admin/safety", "Safety"],
        ["/admin/history", "History"],
        ["/admin/device", "Device"],
      ]
    : [
        ["/home", "Home / Live"],
        ["/assist", "Assist"],
        ["/navigate", "Navigate"],
        ["/safety", "Safety"],
      ];
  return (
    <>
      <header>
        <Link className="brand" to={admin ? "/admin" : "/home"}>
          <Eye />
          VISIONMATE{admin && <small>Caregiver</small>}
        </Link>
        <nav aria-label="Main navigation">
          {links.map(([to, label]) => (
            <NavLink key={to} to={to} end>
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="header-tools">
          <Badge>{online ? "Backend online" : "Backend offline"}</Badge>
          <Link aria-label="Settings" to="/settings">
            <SettingsIcon size={20} />
          </Link>
          <Link to="/" onClick={() => sessionStorage.removeItem("vm-role")}>
            Switch role
          </Link>
        </div>
      </header>
      <div className="connection-line">
        Events: {connection}
        {updated && (
          <span>Last sync {new Date(updated).toLocaleTimeString()}</span>
        )}
      </div>
      {error && (
        <div className="offline" role="alert">
          {error} Displayed data may be stale.
        </div>
      )}
      <main>
        <Outlet />
      </main>
      <footer>
        <span>VisionMate / Drishti · Assistance from your device</span>
        <Action
          danger
          label="STOP all activity"
          run={() => api.action("/api/v1/stop")}
        />
        <OctagonX aria-hidden="true" />
      </footer>
    </>
  );
}
