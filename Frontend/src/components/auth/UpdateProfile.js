import React, { useRef, useState } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { Link, useNavigate } from "react-router-dom";
import "./Auth.css";

export default function UpdateProfile() {
  const emailRef = useRef();
  const passwordRef = useRef();
  const passwordConfirmRef = useRef();
  const { currentUser, updatePassword, updateEmail } = useAuth();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const history = useNavigate();

  function handleSubmit(e) {
    e.preventDefault();
    if (passwordRef.current.value !== passwordConfirmRef.current.value) {
      return setError("Passwords do not match");
    }

    const promises = [];
    setLoading(true);
    setError("");

    if (emailRef.current.value !== currentUser.email) {
      promises.push(updateEmail(emailRef.current.value));
    }
    if (passwordRef.current.value) {
      promises.push(updatePassword(passwordRef.current.value));
    }

    Promise.all(promises)
      .then(() => {
        history("/");
      })
      .catch(() => {
        setError("Failed to update account");
      })
      .finally(() => {
        setLoading(false);
      });
  }

  return (
    <>
      <div className="card ml-64">
        <div className="card-body">
          <h2 className="text-center mb-4">Update Profile</h2>
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
                defaultValue={currentUser.email}
              ></input>
            </div>
            <div className="form-group" id="password">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                className="form-control"
                id="password"
                ref={passwordRef}
                placeholder="Leave blank to keep the same"
              ></input>
            </div>
            <div className="form-group" id="password-confirm">
              <label htmlFor="password-confirm">Password Confirmation</label>
              <input
                type="password"
                className="form-control"
                id="password-confirm"
                ref={passwordConfirmRef}
                placeholder="Leave blank to keep the same"
              ></input>
            </div>
            <button
              disabled={loading}
              type="submit"
              className="btn btn-primary w-100"
            >
              Update
            </button>
          </form>
          <div className="w-100 text-center mt-2">
            <Link to="/">Cancel</Link>
          </div>
        </div>
      </div>
    </>
  );
}
