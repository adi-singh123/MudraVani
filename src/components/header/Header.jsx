import React, { useEffect } from "react";
import "./Header.css";
import SignHand from "../../assests/SignHand.png";

const Header = () => {
  useEffect(() => {
    const logo = document.querySelector(".logo-container");
    
    if (logo) {
      document.addEventListener("mousemove", (e) => {
        const { clientX, clientY } = e;
        const centerX = window.innerWidth / 2;
        const centerY = window.innerHeight / 2;

        // Calculate rotation angles
        const deltaX = (clientX - centerX) / 16;
        const deltaY = (clientY - centerY) / 16;

        logo.style.transform = `rotateX(${deltaY}deg) rotateY(${deltaX}deg)`;
      });
    }

    return () => {
      document.removeEventListener("mousemove", () => {}); // Clean up event listener
    };
  }, []);

  return (
    <div className="signlang__header section__padding gradient__bg" id="home">
      <div className="signlang__header-content">
        <h1 className="gradient__text">Learn and Recognize Indian Sign Language.</h1>
        <p className="text-container">AI-powered real-time sign language recognition</p>
      </div>
      
      <div className="signlang__header-image">
        <img src={SignHand} alt="SIGN-HAND" className="logo-container" />
      </div>
    </div>
  );
};

export default Header;
