import React from "react";
import { Link } from "react-router-dom";
import ContactUs from "../components/ContactUs";
import Services from "../components/Services";

class Home extends React.Component {
  render() {
    return (
      <div>
        <div className="hero" id="home">
          <div className="hero__container">
            <p className="hero__description">
              "Unleash the power of data for your business success"
            </p>
          </div>
        </div>

        <Services />

        <ContactUs />
        <div className="main" id="sign-up">
          <div className="main__container">
            <div className="main__content text-center">
              <h1>Join Our Team</h1>
              <h2>Sign Up Today</h2>
              <p>See what makes us different</p>
              <button className="main__btn">
                <Link href="#" to={"/signup"}>
                  Sign Up
                </Link>
              </button>
            </div>
            <div className="main__img--container">
              <div className="main__img--card" id="card-2">
                <i className="fas fa-users"></i>
              </div>
            </div>
          </div>
        </div>
        <div className="footer__container">
          <div className="footer__links">
            <div className="footer__link--wrapper">
              <div className="footer__link--items">
                <h2>About Us</h2>
                <a href="/sign__up">How it works</a>{" "}
                <a href="/">Testimonials</a>
                <a href="/">Careers</a> <a href="/">Terms of Service</a>
              </div>
              <div className="footer__link--items">
                <h2>Contact Us</h2>
                <a href="/">Contact</a> <a href="/">Support</a>
                <a href="/">Destinations</a>
              </div>
            </div>
            <div className="footer__link--wrapper">
              <div className="footer__link--items">
                <h2>Social Media</h2>
                <a href="/">Instagram</a> <a href="/">Facebook</a>
                <a href="/">Youtube</a> <a href="/">Twitter</a>
              </div>
            </div>
          </div>
          <section className="social__media">
            <div className="social__media--wrap">
              <div className="footer__logo">
                <a href="/" id="footer__logo">
                  Metriverse
                </a>
              </div>
              <p className="website__rights">
                © Metriverse 2023. All rights reserved
              </p>
              <div className="social__icons">
                <a href="/" className="social__icon--link" target="_blank">
                  <i className="fab fa-facebook"></i>
                </a>
                <a href="/" className="social__icon--link">
                  <i className="fab fa-instagram"></i>
                </a>
                <a href="/" className="social__icon--link">
                  <i className="fab fa-youtube"></i>
                </a>
                <a href="/" className="social__icon--link">
                  <i className="fab fa-linkedin"></i>
                </a>
                <a href="/" className="social__icon--link">
                  <i className="fab fa-twitter"></i>
                </a>
              </div>
            </div>
          </section>
        </div>
      </div>
    );
  }
}

export default Home;
