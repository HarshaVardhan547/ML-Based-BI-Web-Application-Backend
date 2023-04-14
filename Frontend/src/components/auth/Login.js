import React, { useRef, useState } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { Link, useNavigate } from "react-router-dom";
import "./Auth.css";

export default function Login() {
  const emailRef = useRef();
  const passwordRef = useRef();
  const { login } = useAuth();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const history = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();

    try {
      setError("");
      setLoading(true);
      await login(emailRef.current.value, passwordRef.current.value);
      history("/");
    } catch {
      setError("Failed to log in");
    }

    setLoading(false);
  }

  return (
    <>
      <div className="card">
        <div className="card-body">
          <h2 className="text-center mb-4">Log In</h2>
          {error && (
            <div className="alert alert-primary" variant="danger">
              {error}
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
            <div className="form-group" id="password">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                className="form-control"
                id="password"
                ref={passwordRef}
                required
              ></input>
            </div>
            <button
              disabled={loading}
              type="submit"
              className="btn btn-primary w-100"
            >
              Log In
            </button>
          </form>
          <div className="w-100 text-center mt-3">
            <Link to="/forgot-password">Forgot Password?</Link>
          </div>
          <div className="w-100 text-center mt-2">
            Need an account? <Link to="/signup">Sign Up</Link>
          </div>
        </div>
      </div>
    </>
  );
}
