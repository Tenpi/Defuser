import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, FormatContext,
SiteLightnessContext, SocketContext, ImageBrightnessContext, ImageContrastContext, TrainStartedContext,
TrainCompletedContext, PreviewImageContext, UpdateImagesContext, SimplifyImageInputContext} from "../Context"
import functions from "../structures/Functions"
import imgPlaceHolder from "../assets/images/img-placeholder.png"
import xIcon from "../assets/icons/x-alt.png"
import "./styles/traintag.less"
import fileType from "magic-bytes.js"
import axios from "axios"
import path from "path"

let timer = null as any
let clicking = false

const SimplifySketch: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {socket, setSocket} = useContext(SocketContext)
    const {format, setFormat} = useContext(FormatContext)
    const {previewImage, setPreviewImage} = useContext(PreviewImageContext)
    const {updateImages, setUpdateImages} = useContext(UpdateImagesContext)
    const {trainStarted, setTrainStarted} = useContext(TrainStartedContext)
    const {trainCompleted, setTrainCompleted} = useContext(TrainCompletedContext)
    const {simplifyImageInput, setSimplifyImageInput} = useContext(SimplifyImageInputContext)
    const [hover, setHover] = useState(false)
    const [img, setImg] = useState(null) as any
    const [output, setOutput] = useState("")
    const [started, setStarted] = useState(false)
    const progressBarRef = useRef(null) as React.RefObject<HTMLDivElement>
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedImage = localStorage.getItem("simplifyImageInput")
        if (savedImage) setSimplifyImageInput(savedImage)
    }, [])

    useEffect(() => {
        try {
            localStorage.setItem("simplifyImageInput", String(simplifyImageInput))
        } catch {
            // ignore
        }
    }, [simplifyImageInput])

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
                        setSimplifyImageInput(link)
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
        setSimplifyImageInput("")
        setImg(null)
        setOutput("")
    }

    const loadImages = async () => {
        if (!simplifyImageInput) return
        const image = document.createElement("img")
        await new Promise<void>((resolve) => {
            image.onload = () => resolve()
            image.src = simplifyImageInput
        })
        setImg(image)
    }
    useEffect(() => {
        loadImages()
    }, [simplifyImageInput])

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

    const simplify = async () => {
        const json = {} as any
        json.image = functions.cleanBase64(simplifyImageInput)
        json.format = format
        await axios.post("/simplify-sketch", json)
    }

    const interrupt = async () => {
        axios.post("/interrupt-misc")
    }

    return (
        <div className="train-tag-row" onMouseEnter={() => setEnableDrag(false)}>
            <div className="train-tag-column" style={{marginRight: "100px"}}>
                <div className="options-bar-img-input">
                    <span className="options-bar-text">Img Input</span>
                    <label htmlFor="img" className="options-bar-img-container" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
                        <div className={`options-bar-img-button-container ${simplifyImageInput && hover ? "show-options-bar-img-buttons" : ""}`}>
                            <img className="options-bar-img-button" src={xIcon} onClick={removeImage} style={{filter: getFilter()}} draggable={false}/>
                        </div>
                        {simplifyImageInput ? 
                        <canvas ref={ref} className="options-bar-img" draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`}}></canvas> :
                        <img className="options-bar-img" src={imgPlaceHolder} style={{filter: getFilter()}} draggable={false}/>}
                    </label>
                    <input id="img" type="file" onChange={(event) => loadImage(event)}/>
                </div>
                <button className="train-tag-button" onClick={() => trainStarted ? interrupt() : simplify()} style={{backgroundColor: trainStarted ? "var(--buttonBGStop)" : "var(--buttonBG)", marginLeft: "0px"}}>{trainStarted ? "Stop" : "Simplify"}</button>
            </div>
            {output ? <div className="train-tag-column">
                <img className="options-bar-img" src={output} style={{filter: getFilter()}} draggable={false} onClick={handleClick}/>
            </div> : null}
        </div>
    )
}

export default SimplifySketch