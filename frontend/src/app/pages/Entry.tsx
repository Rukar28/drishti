import { Link } from "react-router-dom";
import { Eye, ShieldCheck, ArrowRight } from "lucide-react";
export default function Entry() {
  return (
    <main className="entry">
      <Eye size={40} />
      <p className="eyebrow">DRISHTI / VISIONMATE</p>
      <h1>A clearer view. A little more independence.</h1>
      <p className="muted">Choose your experience on this device.</p>
      <div className="grid">
        <Link
          className="panel entry-link"
          to="/home"
          onClick={() => sessionStorage.setItem("vm-role", "user")}
        >
          <Eye />
          <h2>VisionMate User</h2>
          <p>Live vision, assistance and navigation.</p>
          <ArrowRight />
        </Link>
        <Link
          className="panel entry-link"
          to="/admin"
          onClick={() => sessionStorage.setItem("vm-role", "caregiver")}
        >
          <ShieldCheck />
          <h2>Caregiver</h2>
          <p>Location, safety and device health.</p>
          <ArrowRight />
        </Link>
      </div>
      <p className="muted">
        Local prototype entry. This is not authentication. Use only on a trusted
        device and network.
      </p>
    </main>
  );
}
