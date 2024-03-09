import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, FormatContext,
SiteLightnessContext, SocketContext, ImageBrightnessContext, ImageContrastContext, TrainStartedContext,
TrainCompletedContext, PreviewImageContext, UpdateImagesContext, ThemeContext, ShadeImageInputContext,
ImageHueContext, ImageSaturationContext} from "../Context"
import functions from "../structures/Functions"
import imgPlaceHolder from "../assets/images/img-placeholder.png"
import xIcon from "../assets/icons/x-alt.png"
import radioButtonOff from "../assets/icons/radiobutton-off.png"
import radioButtonOn from "../assets/icons/radiobutton-on.png"
import radioButtonOffLight from "../assets/icons/radiobutton-off-light.png"
import radioButtonOnLight from "../assets/icons/radiobutton-on-light.png"
import "./styles/traintag.less"
import fileType from "magic-bytes.js"
import axios from "axios"
import path from "path"

let timer = null as any
let clicking = false

const ShadeSketch: React.FunctionComponent = (props) => {
    const {theme, setTheme} = useContext(ThemeContext)
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
    const {format, setFormat} = useContext(FormatContext)
    const {previewImage, setPreviewImage} = useContext(PreviewImageContext)
    const {updateImages, setUpdateImages} = useContext(UpdateImagesContext)
    const {trainStarted, setTrainStarted} = useContext(TrainStartedContext)
    const {trainCompleted, setTrainCompleted} = useContext(TrainCompletedContext)
    const {shadeImageInput, setShadeImageInput} = useContext(ShadeImageInputContext)
    const [hover, setHover] = useState(false)
    const [img, setImg] = useState(null) as any
    const [output, setOutput] = useState("")
    const [directionA, setDirectionA] = useState("1")
    const [directionB, setDirectionB] = useState("1")
    const [started, setStarted] = useState(false)
    const progressBarRef = useRef(null) as React.RefObject<HTMLDivElement>
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedImage = localStorage.getItem("shadeImageInput")
        if (savedImage) setShadeImageInput(savedImage)
        const savedDirectionA = localStorage.getItem("directionA")
        if (savedDirectionA) setDirectionA(savedDirectionA)
        const savedDirectionB = localStorage.getItem("directionB")
        if (savedDirectionB) setDirectionB(savedDirectionB)
    }, [])

    useEffect(() => {
        try {
            localStorage.setItem("directionA", String(directionA))
            localStorage.setItem("directionB", String(directionB))
            localStorage.setItem("shadeImageInput", String(shadeImageInput))
        } catch {
            // ignore
        }
    }, [shadeImageInput, directionA, directionB])

    useEffect(() => {
        if (!socket) return
        const startTrain = () => {
            setTrainStarted(true)
            setTrainCompleted(false)
            setOutput("")
        }
        const completeTrain = async (data: any) => {
            setTrainCompleted(true)
            setTrainStarted(false)
            setOutput(data.image)
            setUpdateImages(true)
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
                        setShadeImageInput(link)
                    }, 100)
                }
                resolve()
            }
            fileReader.readAsArrayBuffer(file)
        })
        if (event.target) event.target.value = ""
    }

    const preview = () => {
        if (!trainCompleted && !output) return
        setPreviewImage(output)
    }

    const showInFolder = () => {
        if (!trainCompleted && !output) return
        axios.post("/show-in-folder", {path: output})
    }

    const handleClick = (event: any) => {
        if (previewImage) return clearTimeout(timer)
        if (clicking) {
            clicking = false
            clearTimeout(timer)
            return showInFolder()
        }
        clicking = true
        timer = setTimeout(() => {
            clicking = false
            clearTimeout(timer)
            preview()
        }, 200)
    }

    const removeImage = (event?: any) => {
        event?.preventDefault()
        event?.stopPropagation()
        setShadeImageInput("")
        setImg(null)
        setOutput("")
    }

    const loadImages = async () => {
        if (!shadeImageInput) return
        const image = document.createElement("img")
        await new Promise<void>((resolve) => {
            image.onload = () => resolve()
            image.src = shadeImageInput
        })
        setImg(image)
    }
    useEffect(() => {
        loadImages()
    }, [shadeImageInput])

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

    const getDirection = () => {
        if (directionA === "0") {
            if (directionB === "3") {
                return "001"
            } else {
                return "002"
            }
        } else {
            return directionA + directionB + "0"
        }
    }

    const shade = async () => {
        const json = {} as any
        json.image = functions.cleanBase64(shadeImageInput)
        json.format = format
        json.direction = getDirection()
        await axios.post("/shade-sketch", json)
    }

    const interrupt = async () => {
        axios.post("/interrupt-misc")
    }

    const getRadioButton = (condition: boolean) => {
        if (theme === "light") {
            return condition ? radioButtonOnLight : radioButtonOffLight
        } else {
            return condition ? radioButtonOn : radioButtonOff
        }
    }

    return (
        <div className="train-tag-row" onMouseEnter={() => setEnableDrag(false)}>
            <div className="train-tag-column" style={{marginRight: "100px"}}>
                <div className="options-bar-img-input">
                    <span className="options-bar-text">Img Input</span>
                    <label htmlFor="img" className="options-bar-img-container" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
                        <div className={`options-bar-img-button-container ${shadeImageInput && hover ? "show-options-bar-img-buttons" : ""}`}>
                            <img className="options-bar-img-button" src={xIcon} onClick={removeImage} style={{filter: getFilter()}} draggable={false}/>
                        </div>
                        {shadeImageInput ? 
                        <canvas ref={ref} className="options-bar-img" draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%) hue-rotate(${imageHue - 180}deg) saturate(${imageSaturation}%)`}}></canvas> :
                        <img className="options-bar-img" src={imgPlaceHolder} style={{filter: getFilter()}} draggable={false}/>}
                    </label>
                    <input id="img" type="file" onChange={(event) => loadImage(event)}/>
                </div>
                <button className="train-tag-button" onClick={() => trainStarted ? interrupt() : shade()} style={{backgroundColor: trainStarted ? "var(--buttonBGStop)" : "var(--buttonBG)", marginLeft: "0px"}}>{trainStarted ? "Stop" : "Shade"}</button>
            </div>
            <div className="train-tag-column" style={{justifyContent: "center"}}>
                <div className="shade-sketch-box">
                    <div className="shade-sketch-box-row">
                        <img className="shade-sketch-radio-button" src={getRadioButton(directionA === "8")} onClick={() => setDirectionA("8")} style={{filter: getFilter()}}/>
                        <img className="shade-sketch-radio-button" src={getRadioButton(directionA === "1")} onClick={() => setDirectionA("1")} style={{filter: getFilter()}}/>
                        <img className="shade-sketch-radio-button" src={getRadioButton(directionA === "2")} onClick={() => setDirectionA("2")} style={{filter: getFilter()}}/>
                    </div>
                    <div className="shade-sketch-box-row">
                        <img className="shade-sketch-radio-button" src={getRadioButton(directionA === "7")} onClick={() => setDirectionA("7")} style={{filter: getFilter()}}/>
                        <img className="shade-sketch-radio-button" src={getRadioButton(directionA === "0")} onClick={() => setDirectionA("0")} style={{filter: getFilter()}}/>
                        <img className="shade-sketch-radio-button" src={getRadioButton(directionA === "3")} onClick={() => setDirectionA("3")} style={{filter: getFilter()}}/>
                    </div>
                    <div className="shade-sketch-box-row">
                        <img className="shade-sketch-radio-button" src={getRadioButton(directionA === "6")} onClick={() => setDirectionA("6")} style={{filter: getFilter()}}/>
                        <img className="shade-sketch-radio-button" src={getRadioButton(directionA === "5")} onClick={() => setDirectionA("5")} style={{filter: getFilter()}}/>
                        <img className="shade-sketch-radio-button" src={getRadioButton(directionA === "4")} onClick={() => setDirectionA("4")} style={{filter: getFilter()}}/>
                    </div>
                </div>
                <div className="shade-sketch-box">
                    <div className="shade-sketch-box-row">
                        <button className="shade-sketch-button" style={{backgroundColor: directionB === "1" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setDirectionB("1")}>Front</button>
                        <button className="shade-sketch-button" style={{backgroundColor: directionB === "2" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setDirectionB("2")}>Side</button>
                        <button className="shade-sketch-button" style={{backgroundColor: directionB === "3" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setDirectionB("3")}>Back</button>
                    </div>
                </div>
                {output ? <img className="options-bar-img" src={output} style={{filter: getFilter()}} draggable={false} onClick={handleClick}/> : null}
            </div>
        </div>
    )
}

export default ShadeSketch