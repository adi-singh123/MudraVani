import React, { useState, useEffect } from "react";
import "./Navbar.css";
import { Link, useNavigate } from "react-router-dom";
import logo from "../../assests/logo2.png";
import { RiMenu3Line, RiCloseLine } from "react-icons/ri";

const Navbar = () => {
  const [toggle, setToggle] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const navigate = useNavigate();

  // Check login status on component mount
  useEffect(() => {
    const loggedInStatus = localStorage.getItem("isLoggedIn");
    setIsLoggedIn(loggedInStatus === "true");
  }, []);

  const handleLogin = () => {
    navigate("http://localhost:3001/loginage")
  };

  const handleRegister = () => {
    navigate("/register"); // Redirect to register page
  };

  const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    setIsLoggedIn(false);
    navigate("/");
  };

  return (
    <div className="signlang_navbar gradient__bg">
      <div className="singlang_navlinks">
        <div className="signlang_navlinks_logo">
          <Link to="/">
            <img className="logo" src={logo} alt="logo" />
          </Link>
        </div>
        <div className="name">
          <h3>MudraVani</h3>
        </div>

        <div className="signlang_navlinks_container">
          <p className="opp"><Link to="/">Home</Link></p>
          <p className="opp"><Link to="/">Recognize</Link></p>
        </div>

        <div className="signlang_auth-data">
          {isLoggedIn ? (
            <button type="button" onClick={handleLogout}>Logout</button>
          ) : (
            <>
              <button type="button" onClick={handleLogin}>Login</button>
              <button type="button" onClick={handleRegister}>Register</button>
            </>
          )}
        </div>
      </div>

      {/* Mobile Menu */}
      <div className="signlang__navbar-menu">
        {toggle ? (
          <RiCloseLine color="#fff" size={27} onClick={() => setToggle(false)} />
        ) : (
          <RiMenu3Line color="#fff" size={27} onClick={() => setToggle(true)} />
        )}
        {toggle && (
          <div className="signlang__navbar-menu_container scale-up-center">
            <div className="signlang__navbar-menu_container-links">
              <p><Link to="/">Home</Link></p>
              <p><Link to="/detect">Detect</Link></p>
              {isLoggedIn && <p><Link to="/dashboard">Dashboard</Link></p>}
            </div>

            <div className="signlang__navbar-menu_container-links-authdata">
            
                  <Link to={"/login"}><button type="button">Login</button></Link>
                  <button type="button" onClick={handleRegister}>Register</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Navbar;
