import React from "react";
import emailjs from "@emailjs/browser";

function ContactUs() {
  const [yourName, setYourName] = React.useState("");
  const [yourEmail, setYourEmail] = React.useState("");
  const [yourPhone, setYourPhone] = React.useState("");
  const [message, setMessage] = React.useState("");
  function sendEmail() {
    let templateParams = {
      from_name: yourName,
      from_email: yourEmail,
      from_phone: yourPhone,
      message: message,
    };
    emailjs
      .send(
        process.env.REACT_APP_EMAILJS_SERVICE_ID,
        process.env.REACT_APP_EMAILJS_TEMPLATE_ID,
        templateParams,
        process.env.REACT_APP_EMAILJS_PUBLIC_KEY
      )
      .then(
        function (response) {
          console.log("SUCCESS!", response.status, response.text);
          alert("Message Sent, We will get back to you shortly");
        },
        function (error) {
          console.log("FAILED...", error);
        }
      );
  }
  return (
    <div className="main flex justify-center" id="about">
      <div className="container">
        <div className="contact-box">
          <div className="left"></div>
          <div className="right">
            <h2>Contact Us</h2>
            <input
              type="text"
              className="field"
              placeholder="Your Name"
              onChange={(event) => {
                setYourName(event.target.value);
              }}
            />
            <input
              type="text"
              className="field"
              placeholder="Your Email"
              onChange={(event) => {
                setYourEmail(event.target.value);
              }}
            />
            <input
              type="text"
              className="field"
              placeholder="Phone"
              onChange={(event) => {
                setYourPhone(event.target.value);
              }}
            />
            <textarea
              placeholder="Message"
              className="field"
              onChange={(event) => {
                setMessage(event.target.value);
              }}
            ></textarea>
            <button className="main__btn" onClick={sendEmail}>
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ContactUs;
