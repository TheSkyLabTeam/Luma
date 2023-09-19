"use client"
import ControlPanel from "@/components/ControlPanel"
import { useInView, motion } from "framer-motion"
import { useRef, useEffect} from "react"


const page = () => {

  const body = useRef(null);
  const isInView = useInView(body, {once:true, margin:"-10%"})
  
  const animateText = {
    initial: {y: "100%"},
    open: {y: "0%", transition: {duration: 1, delay: 0.2}}
  }

  return (
    <div className="bg-background w-[100vw] h-[100vh] p-4">
      <header className="w-full h-96 flex flex-col p-4 bg-surface rounded-md">
        <div ref={body} className="lineMask">
          <motion.h1 variants={animateText} initial="initial" animate={isInView ? "open" : ""} className="text-5xl font-semibold text-onbackground transform translate-y-1/2">THE SUN TODAY</motion.h1>
        </div>
        <div className="lineMask">
          <motion.h5 variants={animateText} initial="initial" animate={isInView ? "open" : ""} className="text-onbackground">19th september 2023</motion.h5>
        </div>
      </header>
      <ControlPanel/>
    </div>
  )
}

export default page