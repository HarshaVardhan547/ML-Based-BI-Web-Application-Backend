import React, { useRef, useState } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { Link } from "react-router-dom";
import "./Auth.css";

export default function ForgotPassword() {
  const emailRef = useRef();
  const { resetPassword } = useAuth();
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();

    try {
      setMessage("");
      setError("");
      setLoading(true);
      await resetPassword(emailRef.current.value);
      setMessage("Check your inbox for further instructions");
    } catch {
      setError("Failed to reset password");
    }

    setLoading(false);
  }

  return (
    <>
      <div className="card">
        <div className="card-body">
          <h2 className="text-center mb-4">Password Reset</h2>
          {error && (
            <div className="alert alert-primary" variant="danger">
              {error}
            </div>
          )}
          {message && (
            <div className="alert alert-primary" variant="success">
              {message}
            </div>
          )}
          <form onSubmit={handleSubmit}>
            <div className="form-group" id="email">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                className="form-control"
                id="email"
                ref={emailRef}
                required
              ></input>
            </div>
            <button
              className="btn btn-primary w-100"
              disabled={loading}
              type="submit"
            >
              Reset Password
            </button>
          </form>
          <div className="w-100 text-center mt-3">
            <Link to="/login">Login</Link>
          </div>
          <div className="w-100 text-center mt-2">
            Need an account? <Link to="/signup">Sign Up</Link>
          </div>
        </div>
      </div>
    </>
  );
}
