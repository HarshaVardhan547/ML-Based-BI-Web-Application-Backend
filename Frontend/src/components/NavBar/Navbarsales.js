import React from "react";
import { Link } from "react-router-dom";
import { SidebarData } from "./SidebarData";
import { IconContext } from "react-icons";
import { useLocation } from "react-router-dom";

function Navbarsales() {
  const { pathname } = useLocation();
  if (
    pathname === "/" ||
    pathname === "/login" ||
    pathname === "/signup" ||
    pathname === "/forgot-password"
  )
    return null;
  return (
    <>
      <IconContext.Provider value={{ color: "#fff" }}>
        <div className="!sticky top-[3.8125rem] bg-gray-100 z-40">
          <nav className="">
            <div className="px-1 mx-auto border">
              <div className="flex justify-center px-4">
                <div className="text-color-black mt-4">
                  <div className="text-sky-500 hover:text-blue-600"></div>
                  <div className="md:flex space-x-4 items-center px-4">
                    {SidebarData.map((item) => {
                      return (
                        <Link
                          className={
                            "flex items-center lg:text-lg lg:leading-6 mb-4 font-semibold"
                          }
                          to={item.path}
                        >
                          <span className=""> {item.icon} </span>
                          <span className="ml-2">{item.title}</span>
                        </Link>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </nav>
        </div>
      </IconContext.Provider>
    </>
  );
}

export default Navbarsales;
