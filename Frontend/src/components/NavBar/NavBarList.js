import { useNavigate } from "react-router-dom";
import { HashLink as Link } from "react-router-hash-link";
import { useAuth } from "../../contexts/AuthContext";
import React, { useState, useEffect } from "react";

export default function NavBarList() {
  const navigate = useNavigate();
  const [user, setUser] = useState("Login");
  const { currentUser, logout } = useAuth();

  useEffect(() => {
    if (currentUser) {
      console.log("User is logged in");
      setUser(`Logout of ${currentUser.email}`);
    }
  }, [currentUser]);

  function handleLogin() {
    if (user.match(`Logout`)) {
      logout();
      setUser("Login");
    }
    navigate("/login");
  }
  return (
    <div className="md:flex justify-end items-center md:right-4 md:absolute">
      <div className="md:flex space-x-4 items-center px-4">
        <Link to="/#home" className="py-2 hover:text-blue-900" id="homePage">
          Home
        </Link>
        <Link
          to="/#services"
          className="py-2 hover:text-blue-900"
          id="servicesPage"
        >
          Services
        </Link>
        <Link to="/#about" className="py-2 hover:text-blue-900" id="aboutPage">
          About
        </Link>
      </div>
      <button
        className="text-xs leading-5 font-semibold bg-blue-400  rounded-full py-1 px-3 flex items-center space-x-2 hover:bg-blue-600"
        id="signup"
        onClick={handleLogin}
      >
        {user}
      </button>
    </div>
  );
}
