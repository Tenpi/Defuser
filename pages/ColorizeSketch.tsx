import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, FormatContext,
SiteLightnessContext, SocketContext, ImageBrightnessContext, ImageContrastContext, TrainStartedContext,
TrainCompletedContext, PreviewImageContext, UpdateImagesContext, ColorizeSketchInputContext, ColorizeStyleInputContext,
ImageHueContext, ImageSaturationContext} from "../Context"
import functions from "../structures/Functions"
import imgPlaceHolder from "../assets/images/img-placeholder.png"
import xIcon from "../assets/icons/x-alt.png"
import "./styles/traintag.less"
import fileType from "magic-bytes.js"
import axios from "axios"
import path from "path"

let timer = null as any
let clicking = false

const ColorizeSketch: React.FunctionComponent = (props) => {
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
    const {colorizeSketchInput, setColorizeSketchInput} = useContext(ColorizeSketchInputContext)
    const {colorizeStyleInput, setColorizeStyleInput} = useContext(ColorizeStyleInputContext)
    const [hover, setHover] = useState(false)
    const [styleHover, setStyleHover] = useState(false)
    const [img, setImg] = useState(null) as any
    const [styleImg, setStyleImg] = useState(null) as any
    const [output, setOutput] = useState("")
    const [started, setStarted] = useState(false)
    const progressBarRef = useRef(null) as React.RefObject<HTMLDivElement>
    const ref = useRef<HTMLCanvasElement>(null)
    const styleRef = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedSketch = localStorage.getItem("colorizeSketchInput")
        if (savedSketch) setColorizeSketchInput(savedSketch)
        const savedStyle = localStorage.getItem("colorizeStyleInput")
        if (savedStyle) setColorizeStyleInput(savedStyle)
    }, [])

    useEffect(() => {
        try {
            localStorage.setItem("colorizeSketchInput", String(colorizeSketchInput))
            try {
                localStorage.setItem("colorizeStyleInput", String(colorizeStyleInput))
            } catch {}
        } catch {}
    }, [colorizeSketchInput, colorizeStyleInput])

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
                        setColorizeSketchInput(link)
                    }, 100)
                }
                resolve()
            }
            fileReader.readAsArrayBuffer(file)
        })
        if (event.target) event.target.value = ""
    }

    const loadStyleImage = async (event: any) => {
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
                    removeStyleImage()
                    setTimeout(() => {
                        setColorizeStyleInput(link)
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
        setColorizeSketchInput("")
        setImg(null)
        setOutput("")
    }

    const removeStyleImage = (event?: any) => {
        event?.preventDefault()
        event?.stopPropagation()
        setColorizeStyleInput("")
        setStyleImg(null)
    }

    const loadImages = async () => {
        if (colorizeSketchInput) {
            const image = document.createElement("img")
            await new Promise<void>((resolve) => {
                image.onload = () => resolve()
                image.src = colorizeSketchInput
            })
            setImg(image)
        }
        if (colorizeStyleInput) {
            const styleImage = document.createElement("img")
            await new Promise<void>((resolve) => {
                styleImage.onload = () => resolve()
                styleImage.src = colorizeStyleInput
            })
            setStyleImg(styleImage)
        }
    }
    useEffect(() => {
        loadImages()
    }, [colorizeSketchInput, colorizeStyleInput])

    const getNormalizedDimensions = (img: HTMLImageElement) => {
        let greaterValue = img.width > img.height ? img.width : img.height
        const heightBigger = img.height > img.width
        const ratio = greaterValue / (heightBigger ? 800 : 1200)
        const width = Math.floor(img.width / ratio)
        const height = Math.floor(img.height / ratio)
        return {width, height}
    }

    const updateImage = () => {
        if (ref.current && img) {
            ref.current.width = getNormalizedDimensions(img).width
            ref.current.height = getNormalizedDimensions(img).height
            const ctx = ref.current.getContext("2d")!
            ctx.drawImage(img, 0, 0, ref.current.width, ref.current.height)
        }
        if (styleRef.current && styleImg) {
            styleRef.current.width = getNormalizedDimensions(styleImg).width
            styleRef.current.height = getNormalizedDimensions(styleImg).height
            const styleCtx = styleRef.current.getContext("2d")!
            styleCtx.drawImage(styleImg, 0, 0, styleRef.current.width, styleRef.current.height)
        }
    }

    useEffect(() => {
        updateImage()
    }, [img, styleImg, siteHue, siteSaturation, siteLightness])

    const colorize = async () => {
        const json = {} as any
        json.sketch = functions.cleanBase64(colorizeSketchInput)
        json.style = functions.cleanBase64(colorizeStyleInput)
        json.format = format
        await axios.post("/colorize-sketch", json)
    }

    const interrupt = async () => {
        axios.post("/interrupt-misc")
    }

    return (
        <div className="train-tag-row" onMouseEnter={() => setEnableDrag(false)}>
            <div className="train-tag-column">
                <div className="options-bar-img-input">
                    <span className="options-bar-text">Sketch Input</span>
                    <label htmlFor="img" className="options-bar-img-container" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
                        <div className={`options-bar-img-button-container ${colorizeSketchInput && hover ? "show-options-bar-img-buttons" : ""}`}>
                            <img className="options-bar-img-button" src={xIcon} onClick={removeImage} style={{filter: getFilter()}} draggable={false}/>
                        </div>
                        {colorizeSketchInput ? 
                        <canvas ref={ref} className="options-bar-img" draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`}}></canvas> :
                        <img className="options-bar-img" src={imgPlaceHolder} style={{filter: getFilter()}} draggable={false}/>}
                    </label>
                    <input id="img" type="file" onChange={(event) => loadImage(event)}/>
                </div>
                <button className="train-tag-button" onClick={() => trainStarted ? interrupt() : colorize()} style={{backgroundColor: trainStarted ? "var(--buttonBGStop)" : "var(--buttonBG)", marginLeft: "0px"}}>{trainStarted ? "Stop" : "Colorize"}</button>
            </div>
            <div className="train-tag-column">
                <div className="options-bar-img-input">
                    <span className="options-bar-text">Style Input</span>
                    <label htmlFor="style-img" className="options-bar-img-container" onMouseEnter={() => setStyleHover(true)} onMouseLeave={() => setStyleHover(false)}>
                        <div className={`options-bar-img-button-container ${colorizeStyleInput && styleHover ? "show-options-bar-img-buttons" : ""}`}>
                            <img className="options-bar-img-button" src={xIcon} onClick={removeStyleImage} style={{filter: getFilter()}} draggable={false}/>
                        </div>
                        {colorizeStyleInput ? 
                        <canvas ref={styleRef} className="options-bar-img" draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%) hue-rotate(${imageHue - 180}deg) saturate(${imageSaturation}%)`}}></canvas> :
                        <img className="options-bar-img" src={imgPlaceHolder} style={{filter: getFilter()}} draggable={false}/>}
                    </label>
                    <input id="style-img" type="file" onChange={(event) => loadStyleImage(event)}/>
                </div>
            </div>
            {output ? <div className="train-tag-column" style={{justifyContent: "center"}}>
                <img className="options-bar-img" src={output} style={{filter: getFilter()}} draggable={false} onClick={handleClick}/>
            </div> : null}
        </div>
    )
}

export default ColorizeSketch