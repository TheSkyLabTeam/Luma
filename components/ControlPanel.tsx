import { motion } from "framer-motion"
import { ModeSelector } from "./ModeSelector"
import { DatePicker } from "./ui/DatePicker"

const ControlPanel = () => {

  return (
    <div  id="controlPanel" className="w-fit h-14 p-2 bg-surface flex items-center absolute bottom-4 rounded-lg gap-2 left-1/2 transform -translate-x-1/2">
        <ModeSelector/>
        <DatePicker/>
    </div>
  )
}

export default ControlPanel