import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, 
SiteLightnessContext, SocketContext, ImageBrightnessContext, ImageContrastContext,
TrainStartedContext, TrainCompletedContext, AIImageInputContext, ImageHueContext,
ImageSaturationContext, ThemeContext, ThemeSelectorContext} from "../Context"
import functions from "../structures/Functions"
import imgPlaceHolder from "../assets/images/img-placeholder.png"
import xIcon from "../assets/icons/x-alt.png"
import "./styles/traintag.less"
import fileType from "magic-bytes.js"
import axios from "axios"
import path from "path"

const AIDetector: React.FunctionComponent = (props) => {
    const {theme, setTheme} = useContext(ThemeContext)
    const {themeSelector, setThemeSelector} = useContext(ThemeSelectorContext)
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {imageHue, setImageHue} = useContext(ImageHueContext)
    const {imageSaturation, setImageSaturation} = useContext(ImageSaturationContext)
    const {socket, setSocket} = useContext(SocketContext)
    const {trainStarted, setTrainStarted} = useContext(TrainStartedContext)
    const {trainCompleted, setTrainCompleted} = useContext(TrainCompletedContext)
    const {aiImageInput, setAIImageInput} = useContext(AIImageInputContext)
    const [hover, setHover] = useState(false)
    const [img, setImg] = useState(null) as any
    const [label, setLabel] = useState("")
    const [probability, setProbability] = useState(0)
    const [started, setStarted] = useState(false)
    const progressBarRef = useRef(null) as React.RefObject<HTMLDivElement>
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        let saturation = siteSaturation
        let lightness = siteLightness
        if (themeSelector === "original") {
            if (theme === "light") saturation -= 60
            if (theme === "light") lightness += 90
        } else if (themeSelector === "accessibility") {
            if (theme === "light") saturation -= 90
            if (theme === "light") lightness += 200
            if (theme === "dark") saturation -= 50
            if (theme === "dark") lightness -= 30
        }
        return `hue-rotate(${siteHue - 180}deg) saturate(${saturation}%) brightness(${lightness + 50}%)`
    }

    useEffect(() => {
        const savedImage = localStorage.getItem("aiImageInput")
        if (savedImage) setAIImageInput(savedImage)
    }, [])

    useEffect(() => {
        try {
            localStorage.setItem("aiImageInput", String(aiImageInput))
        } catch {
            // ignore
        }
    }, [aiImageInput])

    useEffect(() => {
        if (!socket) return
        const startTrain = () => {
            setTrainStarted(true)
            setTrainCompleted(false)
        }
        const completeTrain = async (data: any) => {
            setTrainCompleted(true)
            setTrainStarted(false)
            setLabel(data.label)
            setProbability(data.probability)
        }
        const interruptTrain = () => {
            setTrainStarted(false)
        }
        socket.on("train starting", startTrain)
        socket.on("train complete", completeTrain)
        socket.on("train interrupt", interruptTrain)
        return () => {
            socket.off("train starting", startTrain)
            socket.off("train complete", completeTrain)
            socket.off("train interrupt", interruptTrain)
        }
    }, [socket])

    const loadImage = async (event: any) => {
        const file = event.target.files?.[0]
        if (!file) return
        const fileReader = new FileReader()
        await new Promise<void>((resolve) => {
            fileReader.onloadend = async (f: any) => {
                let bytes = new Uint8Array(f.target.result)
                const result = fileType(bytes)?.[0]
                const jpg = result?.mime === "image/jpeg" ||
                path.extname(file.name).toLowerCase() === ".jpg" ||
                path.extname(file.name).toLowerCase() === ".jpeg"
                const png = result?.mime === "image/png"
                const webp = result?.mime === "image/webp"
                if (jpg) result.typename = "jpg"
                if (jpg || png || webp) {
                    const url = functions.arrayBufferToBase64(bytes.buffer)
                    const link = `${url}#.${result.typename}`
                    removeImage()
                    setTimeout(() => {
                        setAIImageInput(link)
                    }, 100)
                }
                resolve()
            }
            fileReader.readAsArrayBuffer(file)
        })
        if (event.target) event.target.value = ""
    }

    const removeImage = (event?: any) => {
        event?.preventDefault()
        event?.stopPropagation()
        setAIImageInput("")
        setImg(null)
        setLabel("")
        setProbability(0)
    }

    const loadImages = async () => {
        if (!aiImageInput) return
        const image = document.createElement("img")
        await new Promise<void>((resolve) => {
            image.onload = () => resolve()
            image.src = aiImageInput
        })
        setImg(image)
    }
    useEffect(() => {
        loadImages()
    }, [aiImageInput])

    const getNormalizedDimensions = () => {
        let greaterValue = img.width > img.height ? img.width : img.height
        const heightBigger = img.height > img.width
        const ratio = greaterValue / (heightBigger ? 800 : 1200)
        const width = Math.floor(img.width / ratio)
        const height = Math.floor(img.height / ratio)
        return {width, height}
    }

    const updateImage = () => {
        if (!ref.current || !img) return
        ref.current.width = getNormalizedDimensions().width
        ref.current.height = getNormalizedDimensions().height
        const ctx = ref.current.getContext("2d")!
        ctx.drawImage(img, 0, 0, ref.current.width, ref.current.height)
    }

    useEffect(() => {
        updateImage()
    }, [img, siteHue, siteSaturation, siteLightness])

    const detect = async () => {
        const json = {} as any
        json.image = functions.cleanBase64(aiImageInput)
        await axios.post("/ai-detector", json)
    }

    const interrupt = async () => {
        axios.post("/interrupt-misc")
    }

    const getLabel = () => {
        if (label === "ai") return "AI"
        if (label === "human") return "Human"
    }

    return (
        <div className="train-tag" onMouseEnter={() => setEnableDrag(false)}>
            <div className="train-tag-column">
                <div className="options-bar-img-input">
                    <span className="options-bar-text">Img Input</span>
                    <label htmlFor="img" className="options-bar-img-container" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
                        <div className={`options-bar-img-button-container ${aiImageInput && hover ? "show-options-bar-img-buttons" : ""}`}>
                            <img className="options-bar-img-button" src={xIcon} onClick={removeImage} style={{filter: getFilter()}} draggable={false}/>
                        </div>
                        {aiImageInput ? 
                        <canvas ref={ref} className="options-bar-img" draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%) hue-rotate(${imageHue - 180}deg) saturate(${imageSaturation}%)`}}></canvas> :
                        <img className="options-bar-img" src={imgPlaceHolder} style={{filter: getFilter()}} draggable={false}/>}
                    </label>
                    <input id="img" type="file" onChange={(event) => loadImage(event)}/>
                </div>
                <button className="train-tag-button" onClick={() => trainStarted ? interrupt() : detect()} style={{backgroundColor: trainStarted ? "var(--buttonBGStop)" : "var(--buttonBG)", marginLeft: "0px"}}>{trainStarted ? "Stop" : "Detect"}</button>
                {label ? <span className="train-tag-settings-title">Result: {getLabel()}</span> : null}
                {probability ? <span className="train-tag-settings-title">Confidence: {probability}%</span> : null}
            </div>
        </div>
    )
}

export default AIDetector