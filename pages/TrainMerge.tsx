import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
SocketContext, ModelNamesContext, TrainStartedContext, TrainProgressContext, TrainCompletedContext, TrainProgressTextContext,
ThemeContext} from "../Context"
import {ProgressBar, Dropdown, DropdownButton} from "react-bootstrap"
import functions from "../structures/Functions"
import $1 from "../assets/icons/1.png"
import $2 from "../assets/icons/2.png"
import $3 from "../assets/icons/3.png"
import "./styles/traintag.less"
import axios from "axios"

const TrainMerge: React.FunctionComponent = (props) => {
    const {theme, setTheme} = useContext(ThemeContext)
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {socket, setSocket} = useContext(SocketContext)
    const {trainProgress, setTrainProgress} = useContext(TrainProgressContext)
    const {trainProgressText, setTrainProgressText} = useContext(TrainProgressTextContext)
    const {trainStarted, setTrainStarted} = useContext(TrainStartedContext)
    const {trainCompleted, setTrainCompleted} = useContext(TrainCompletedContext)
    const {modelNames, setModelNames} = useContext(ModelNamesContext)
    const [model1, setModel1] = useState("")
    const [model2, setModel2] = useState("")
    const [model3, setModel3] = useState("")
    const [alpha, setAlpha] = useState("0.5")
    const [interpolation, setInterpolation] = useState("weighted_sum")
    const progressBarRef = useRef(null) as React.RefObject<HTMLDivElement>
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedModel1 = localStorage.getItem("mergeModel1")
        if (savedModel1) setModel1(savedModel1)
        const savedModel2 = localStorage.getItem("mergeModel2")
        if (savedModel2) setModel2(savedModel2)
        const savedModel3 = localStorage.getItem("mergeModel3")
        if (savedModel3) setModel3(savedModel3)
        const savedAlpha = localStorage.getItem("alpha")
        if (savedAlpha) setAlpha(savedAlpha)
        const savedInterpolation = localStorage.getItem("interpolation")
        if (savedInterpolation) setInterpolation(savedInterpolation)
    }, [])

    useEffect(() => {
        localStorage.setItem("mergeModel1", String(model1))
        localStorage.setItem("mergeModel2", String(model2))
        localStorage.setItem("mergeModel3", String(model3))
        localStorage.setItem("alpha", String(alpha))
        localStorage.setItem("interpolation", String(interpolation))
    }, [model1, model2, model3, alpha, interpolation])

    useEffect(() => {
        if (!socket) return
        const startTrain = () => {
            setTrainStarted(true)
            setTrainCompleted(false)
            setTrainProgress(-1)
            setTrainProgressText("")
        }
        const trackProgress = (data: any) => {
            const progress = (100 / Number(data.total_step)) * Number(data.step)
            setTrainStarted(true)
            setTrainCompleted(false)
            setTrainProgress(progress)
            setTrainProgressText(`${data.step} / ${data.total_step}`)
        }
        const completeTrain = async (data: any) => {
            setTrainCompleted(true)
            setTrainStarted(false)
        }
        const interruptTrain = () => {
            setTrainStarted(false)
        }
        socket.on("train starting", startTrain)
        socket.on("train progress", trackProgress)
        socket.on("train complete", completeTrain)
        socket.on("train interrupt", interruptTrain)
        return () => {
            socket.off("train starting", startTrain)
            socket.off("train progress", trackProgress)
            socket.off("train complete", completeTrain)
            socket.off("train interrupt", interruptTrain)
        }
    }, [socket])

    const model1JSX = () => {
        let jsx = [] as any
        const items = ["", ...modelNames]
        for (let i = 0; i < items.length; i++) {
            jsx.push(<Dropdown.Item active={model1 === items[i]} onClick={() => setModel1(items[i])}>{items[i] ? items[i] : "None"}</Dropdown.Item>)
        }
        return jsx 
    }

    const model2JSX = () => {
        let jsx = [] as any
        const items = ["", ...modelNames]
        for (let i = 0; i < items.length; i++) {
            jsx.push(<Dropdown.Item active={model2 === items[i]} onClick={() => setModel2(items[i])}>{items[i] ? items[i] : "None"}</Dropdown.Item>)
        }
        return jsx 
    }

    const model3JSX = () => {
        let jsx = [] as any
        const items = ["", ...modelNames]
        for (let i = 0; i < items.length; i++) {
            jsx.push(<Dropdown.Item active={model3 === items[i]} onClick={() => setModel3(items[i])}>{items[i] ? items[i] : "None"}</Dropdown.Item>)
        }
        return jsx 
    }

    const getText = () => {
        if (trainCompleted) return "Completed"
        if (trainProgress >= 0) return trainProgressText
        return "Starting"
    }

    const getProgress = () => {
        if (trainCompleted) return 100
        if (trainProgress >= 0) return trainProgress
        return 0
    }

    const merge = async () => {
        const json = {} as any
        json.models = [model1, model2, model3].filter(Boolean)
        json.alpha = Number(alpha)
        json.interpolation = interpolation
        await axios.post("/merge", json)
    }

    const interruptTrain = async () => {
        axios.post("/interrupt-train")
    }

    const openFolder = async () => {
        await axios.post("/open-folder", {path: `outputs/models/merged`})
    }

    const reset = () => {
        setModel1("")
        setModel2("")
        setModel3("")
        setAlpha("0.5")
        setInterpolation("weighted_sum")
    }

    const twoModels = () => {
        return [model1, model2, model3].filter(Boolean).length <= 2
    }

    useEffect(() => {
        if (!twoModels()) setInterpolation("add_difference")
    }, [model1, model2, model3])

    return (
        <div className="train-tag" onMouseEnter={() => setEnableDrag(false)} style={{height: "86vh"}}>
            <div className="train-tag-checkpoint-bar" onMouseEnter={() => setEnableDrag(false)}>
                <img className="checkpoint-bar-icon" src={$1} style={{cursor: "default", filter: getFilter()}}/>
                <DropdownButton title={model1 ? model1 : "None"} drop="down" className="checkpoint-selector" style={{marginRight: "10px"}}>
                    {model1JSX()}
                </DropdownButton>
                <img className="checkpoint-bar-icon" src={$2} style={{cursor: "default", filter: getFilter()}}/>
                <DropdownButton title={model2 ? model2 : "None"} drop="down" className="checkpoint-selector" style={{marginRight: "10px"}}>
                    {model2JSX()}
                </DropdownButton>
                <img className="checkpoint-bar-icon" src={$3} style={{cursor: "default", filter: getFilter()}}/>
                <DropdownButton title={model3 ? model3 : "None"} drop="down" className="checkpoint-selector">
                    {model3JSX()}
                </DropdownButton>
            </div>
            <div className="train-tag-settings-container">
                <div className="train-tag-settings-column">
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Alpha:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={alpha} onChange={(event) => setAlpha(event.target.value)}/>
                    </div>
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Interpolation:</span>
                        <DropdownButton title={interpolation.replaceAll("_", " ")} drop="down" className="checkpoint-selector">
                            {twoModels() ? <Dropdown.Item active={interpolation === "weighted_sum"} onClick={() => setInterpolation("weighted_sum")}>weighted sum</Dropdown.Item> : null}
                            <Dropdown.Item active={interpolation === "add_difference"} onClick={() => setInterpolation("add_difference")}>add difference</Dropdown.Item>
                            {twoModels() ? <Dropdown.Item active={interpolation === "sigmoid"} onClick={() => setInterpolation("sigmoid")}>sigmoid</Dropdown.Item> : null}
                            {twoModels() ? <Dropdown.Item active={interpolation === "inverse_sigmoid"} onClick={() => setInterpolation("inverse_sigmoid")}>inverse sigmoid</Dropdown.Item> : null}
                        </DropdownButton>
                    </div>
                </div>
            </div>
            {trainStarted ? <div className="train-tag-progress" style={{justifyContent: "flex-start"}}>
                <div className="render-progress-container" style={{filter: getFilter()}}>
                    <span className="render-progress-text">{getText()}</span>
                    <ProgressBar ref={progressBarRef} animated now={getProgress()}/>
                </div>
            </div> : null}
            <div className="train-tag-folder-container" style={{marginTop: "10px"}}>
                <button className="train-tag-button" onClick={() => trainStarted ? interruptTrain() : merge()} style={{backgroundColor: trainStarted ? "var(--buttonBGStop)" : "var(--buttonBG)"}}>{trainStarted ? "Stop" : "Merge"}</button>
                <button className="train-tag-button" onClick={() => openFolder()}>Open</button>
                <button className="train-tag-button" onClick={() => reset()}>Reset</button>
            </div>
        </div>
    )
}

export default TrainMerge