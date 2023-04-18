import { Link } from "react-router-dom";
export default function Services() {
  return (
    <div className="services" id="services">
      <h1>Our Services</h1>
      <div className="services__wrapper">
        <Link className="link_colour" to="/upload">
          <div className="services__card">
            <img
              src="assets/upload.png"
              width="80%"
              height="80%"
              className="ml-auto mr-auto"
            />
            <h2>Upload your Data</h2>

            <div className="services__btn">
              <button>Get Started</button>
            </div>
          </div>
        </Link>
        <Link className="link_colour" to="/EDA">
          <div className="services__card">
            <img
              src="assets/eda.png"
              width="80%"
              height="80%"
              className="ml-auto mr-auto"
            />
            <h2>Exploratory Data Analysis</h2>
            <div className="services__btn">
              <button>Get Started</button>
            </div>
          </div>
        </Link>
        <Link className="link_colour" to="/Insights">
          <div className="services__card">
            <img
              src="assets/insights.png"
              width="80%"
              height="80%"
              className="ml-auto mr-auto"
            />
            <h2>Insights</h2>
            <div className="services__btn">
              <button>Get Started</button>
            </div>
          </div>
        </Link>
        <Link className="link_colour" to="/DataModeling">
          <div className="services__card">
            <img
              src="assets/individual.png"
              width="80%"
              height="80%"
              className="ml-auto mr-auto"
            />
            <h2>Indvidual Customer Analysis</h2>
            <div className="services__btn">
              <button>Get Started</button>
            </div>
          </div>
        </Link>
      </div>
    </div>
  );
}
