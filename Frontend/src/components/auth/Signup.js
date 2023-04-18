import React, { useRef, useState } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { Link, useNavigate } from "react-router-dom";
import "./Auth.css";

export default function Signup() {
  const emailRef = useRef();
  const passwordRef = useRef();
  const passwordConfirmRef = useRef();
  const { signup } = useAuth();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const history = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();

    if (passwordRef.current.value !== passwordConfirmRef.current.value) {
      return setError("Passwords do not match");
    }

    try {
      setError("");
      setLoading(true);
      await signup(emailRef.current.value, passwordRef.current.value);
      history("/");
    } catch (error) {
      console.log(error);
      setError("Failed to create an account");
    }

    setLoading(false);
  }

  return (
    <>
      <div className="card">
        <div className="card-body">
          <h2 className="text-center mb-4">Sign Up</h2>
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
              ></input>
            </div>
            <div className="form-group" id="password-confirm">
              <label htmlFor="password-confirm">Password Confirmation</label>
              <input
                type="password"
                className="form-control"
                id="password-confirm"
                ref={passwordConfirmRef}
              ></input>
            </div>
            <button
              disabled={loading}
              type="submit"
              className="btn btn-primary w-100"
            >
              Sign Up
            </button>
          </form>
          <div className="w-100 text-center mt-2">
            Already have an account? <Link to="/login">Log In</Link>
          </div>
        </div>
      </div>
    </>
  );
}
