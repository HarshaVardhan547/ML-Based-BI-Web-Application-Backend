import { Link } from "react-router-dom";
import NavBarList from "./NavBarList";

export default function Navigationbar() {
  return (
    <div className="!sticky top-0 bg-gray-100 z-40">
      <nav className="">
        <div className="px-1 mx-auto border">
          <div className="md:flex items-center px-4">
            <Link
              to={"/#home"}
              id="navLogo"
              className={"md:flex items-center mb-4 font-semibold"}
            >
              <img src="assets/Logo.png" className="w-64 h-16" />
            </Link>
            <NavBarList />
          </div>
        </div>
      </nav>
    </div>
  );
}
