import React from "react";
import * as AiIcons from "react-icons/ai";
import * as MdIcons from "react-icons/md";
import { IconContext } from "react-icons";

export const SidebarData = [
  {
    title: "Upload",
    path: "/Upload",
    icon: (
      <IconContext.Provider
        value={{ className: "text-sky-500 hover:text-blue-600" }}
      >
        <AiIcons.AiOutlineCloudUpload />
      </IconContext.Provider>
    ),
  },
  {
    title: "Exploratory Data Analysis",
    path: "/EDA",
    icon: (
      <IconContext.Provider
        value={{ className: "text-sky-500 hover:text-blue-600" }}
      >
        <AiIcons.AiOutlineDatabase />
      </IconContext.Provider>
    ),
  },
  {
    title: "Insights",
    path: "/Insights",
    icon: (
      <IconContext.Provider
        value={{ className: "text-sky-500 hover:text-blue-600" }}
      >
        <MdIcons.MdModelTraining />
      </IconContext.Provider>
    ),
  },
  {
    title: "Indvidual Customer Analysis",
    path: "/DataModeling",
    icon: (
      <IconContext.Provider
        value={{ className: "text-sky-500 hover:text-blue-600" }}
      >
        <MdIcons.MdOutlineAnalytics />
      </IconContext.Provider>
    ),
  },
];
