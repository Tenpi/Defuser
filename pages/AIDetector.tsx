import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, 
SiteLightnessContext, SocketContext, ImageBrightnessContext, ImageContrastContext} from "../Context"
import functions from "../structures/Functions"
import imgPlaceHolder from "../assets/images/img-placeholder.png"
import xIcon from "../assets/icons/x-alt.png"
import "./styles/traintag.less"
import fileType from "magic-bytes.js"
import axios from "axios"
import path from "path"

const AIDetector: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {socket, setSocket} = useContext(SocketContext)
    const [imageInput, setImageInput] = useState("")
    const [hover, setHover] = useState(false)
    const [img, setImg] = useState(null) as any
    const [label, setLabel] = useState("")
    const [probability, setProbability] = useState(0)
    const [started, setStarted] = useState(false)
    const progressBarRef = useRef(null) as React.RefObject<HTMLDivElement>
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedImage = localStorage.getItem("aiImageInput")
        if (savedImage) setImageInput(savedImage)
    }, [])

    useEffect(() => {
        try {
            localStorage.setItem("aiImageInput", String(imageInput))
        } catch {
            // ignore
        }
    }, [imageInput])

    useEffect(() => {
        if (!socket) return
        const detectorResult = async (data: any) => {
        }
        socket.on("detector result", detectorResult)
        return () => {
            socket.off("detector result", detectorResult)
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
                        setImageInput(link)
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
        setImageInput("")
        setImg(null)
        setLabel("")
        setProbability(0)
    }

    const loadImages = async () => {
        if (!imageInput) return
        const image = document.createElement("img")
        await new Promise<void>((resolve) => {
            image.onload = () => resolve()
            image.src = imageInput
        })
        setImg(image)
    }
    useEffect(() => {
        loadImages()
    }, [imageInput])

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
        setStarted(true)
        const json = {} as any
        json.image = functions.cleanBase64(imageInput)
        const result = await axios.post("/ai-detector", json).then((r) => r.data)
        setLabel(result.label)
        setProbability(result.probability)
        setStarted(false)
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
                        <div className={`options-bar-img-button-container ${imageInput && hover ? "show-options-bar-img-buttons" : ""}`}>
                            <img className="options-bar-img-button" src={xIcon} onClick={removeImage} style={{filter: getFilter()}} draggable={false}/>
                        </div>
                        {imageInput ? 
                        <canvas ref={ref} className="options-bar-img" draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`}}></canvas> :
                        <img className="options-bar-img" src={imgPlaceHolder} style={{filter: getFilter()}} draggable={false}/>}
                    </label>
                    <input id="img" type="file" onChange={(event) => loadImage(event)}/>
                </div>
                <button className="train-tag-button" onClick={() => started ? null : detect()} style={{backgroundColor: started ? "var(--buttonBGStop)" : "var(--buttonBG)"}}>{started ? "Wait" : "Detect"}</button>
                {label ? <span className="train-tag-settings-title">Result: {getLabel()}</span> : null}
                {probability ? <span className="train-tag-settings-title">Confidence: {probability}%</span> : null}
            </div>
        </div>
    )
}

export default AIDetector