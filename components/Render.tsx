import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {ProgressBar} from "react-bootstrap"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, RenderImageContext, StartedContext,
SocketContext, UpdateImagesContext, PreviewImageContext, ImageBrightnessContext, ImageContrastContext, SidebarTypeContext, AIWatermarkBrightnessContext, 
AIWatermarkHueContext, AIWatermarkInvertContext, AIWatermarkMarginXContext, AIWatermarkMarginYContext, AIWatermarkOpacityContext, AIWatermarkPositionContext, 
AIWatermarkSaturationContext, AIWatermarkScaleContext, AIWatermarkTypeContext, WatermarkContext} from "../Context"
import xIcon from "../assets/icons/x.png"
import xIconHover from "../assets/icons/x-hover.png"
import functions from "../structures/Functions"
import "./styles/render.less"
import ai from "../assets/icons/AI/ai.png"
import aiFan from "../assets/icons/AI/ai-fan.png"
import aiChip from "../assets/icons/AI/ai-chip.png"
import aiPencil from "../assets/icons/AI/ai-pencil.png"
import axios from "axios"

let timer = null as any
let clicking = false

const Render: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {renderImage, setRenderImage} = useContext(RenderImageContext)
    const {previewImage, setPreviewImage} = useContext(PreviewImageContext)
    const {socket, setSocket} = useContext(SocketContext)
    const {updateImages, setUpdateImages} = useContext(UpdateImagesContext)
    const {sidebarType, setSidebarType} = useContext(SidebarTypeContext)
    const [progress, setProgress] = useState(100)
    const {started, setStarted} = useContext(StartedContext)
    const [completed, setCompleted] = useState(false)
    const [upscaling, setUpscaling] = useState(false)
    const [hover, setHover] = useState(false)
    const [xHover, setXHover] = useState(false)
    const {aiWatermarkPosition, setAIWatermarkPosition} = useContext(AIWatermarkPositionContext)
    const {aiWatermarkType, setAIWatermarkType} = useContext(AIWatermarkTypeContext)
    const {aiWatermarkHue, setAIWatermarkHue} = useContext(AIWatermarkHueContext)
    const {aiWatermarkSaturation, setAIWatermarkSaturation} = useContext(AIWatermarkSaturationContext)
    const {aiWatermarkBrightness, setAIWatermarkBrightness} = useContext(AIWatermarkBrightnessContext)
    const {aiWatermarkInvert, setAIWatermarkInvert} = useContext(AIWatermarkInvertContext)
    const {aiWatermarkOpacity, setAIWatermarkOpacity} = useContext(AIWatermarkOpacityContext)
    const {aiWatermarkMarginX, setAIWatermarkMarginX} = useContext(AIWatermarkMarginXContext)
    const {aiWatermarkMarginY, setAIWatermarkMarginY} = useContext(AIWatermarkMarginYContext)
    const {aiWatermarkScale, setAIWatermarkScale} = useContext(AIWatermarkScaleContext)
    const {watermark, setWatermark} = useContext(WatermarkContext)
    const progressBarRef = useRef(null) as React.RefObject<HTMLDivElement>
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    const getWatermarkImage = () => {
        if (aiWatermarkType === "none") return ai
        if (aiWatermarkType === "fan") return aiFan
        if (aiWatermarkType === "chip") return aiChip
        if (aiWatermarkType === "pencil") return aiPencil
        return ai
    }

    useEffect(() => {
        if (!socket) return
        const startingImage = () => {
            setStarted(true)
            setCompleted(false)
            setProgress(-1)
        }
        const stepProgress = (data: any) => {
            const pixels = new Uint8Array(data.image)
            const canvas = document.createElement("canvas")
            canvas.width = data.width
            canvas.height = data.height
            const ctx = canvas.getContext("2d")
            const newImageData = new ImageData(data.width, data.height)
            newImageData.data.set(pixels)
            ctx?.putImageData(newImageData, 0, 0)
            const url = canvas.toDataURL()
            setRenderImage(url)
            const progress = (100 / Number(data.total_step)) * Number(data.step)
            setStarted(true)
            setCompleted(false)
            setProgress(progress)
        }
        const upscalingImage = (data: any) => {
            setUpscaling(true)
        }
        const completedImage = async (data: any) => {
            const newType = functions.getSidebarType(data.image, sidebarType)
            if (newType !== sidebarType) setSidebarType(newType)
            if (data.needs_watermark && watermark) {
                const watermarked = await functions.watermarkImage(data.image, getWatermarkImage(), {aiWatermarkBrightness, aiWatermarkHue, 
                aiWatermarkSaturation, aiWatermarkPosition, aiWatermarkInvert, aiWatermarkMarginX, aiWatermarkMarginY, 
                aiWatermarkOpacity, aiWatermarkType, aiWatermarkScale})
                const form = new FormData()
                const arrayBuffer = await fetch(watermarked).then((r) => r.arrayBuffer())
                const blob = new Blob([new Uint8Array(arrayBuffer)])
                const file = new File([blob], "image.png", {type: "image/png"})
                form.append("image", file)
                form.append("path", data.image)
                await axios.post("/save-watermark", form)
            }
            setCompleted(true)
            setUpscaling(false)
            setStarted(false)
            setRenderImage(data.image ? `${data.image}?v=${new Date().getTime()}` : "")
            setUpdateImages(true)
        }
        const interruptImage = () => {
            setStarted(false)
            setUpscaling(false)
            setRenderImage("")
        }
        socket.on("image starting", startingImage)
        socket.on("step progress", stepProgress)
        socket.on("image upscaling", upscalingImage)
        socket.on("image complete", completedImage)
        socket.on("image interrupt", interruptImage)
        return () => {
            socket.off("image starting", startingImage)
            socket.off("step progress", stepProgress)
            socket.off("image upscaling", upscalingImage)
            socket.off("image complete", completedImage)
            socket.off("image interrupt", interruptImage)
        }
    }, [socket, sidebarType, aiWatermarkPosition, aiWatermarkType, aiWatermarkHue, aiWatermarkSaturation, aiWatermarkBrightness, 
        aiWatermarkInvert, aiWatermarkOpacity, aiWatermarkMarginX, aiWatermarkMarginY,aiWatermarkScale, watermark])

    const updateProgressColor = () => {
        const progressBar = progressBarRef.current?.querySelector(".progress-bar") as HTMLElement
        if (progressBar) progressBar.style.backgroundColor = "#fe0c75"
    }

    useEffect(() => {
        updateProgressColor()
    }, [])

    const getText = () => {
        if (completed) return "Completed"
        if (upscaling) return "Upscaling"
        if (progress >= 0) return `${progress}%`
        return "Starting"
    }

    const getProgress = () => {
        if (completed) return 100
        if (upscaling) return 100
        if (progress >= 0) return progress
        return 0
    }

    const preview = () => {
        if (!completed || !renderImage) return
        setPreviewImage(renderImage)
    }

    const showInFolder = () => {
        if (!completed || !renderImage) return
        axios.post("/show-in-folder", {path: renderImage})
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

    const remove = () => {
        if (!completed) return
        setStarted(false)
        setRenderImage("")
    }

    if (!renderImage && !started) return null

    return (
        <div className="render" onMouseEnter={() => setEnableDrag(false)}>
            <div className="render-progress-container" style={{filter: getFilter()}}>
                <span className="render-progress-text">{getText()}</span>
                <ProgressBar ref={progressBarRef} animated now={getProgress()}/>
            </div>
            {renderImage ? <div className="render-img-container" onClick={handleClick} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
                {completed ? <div className={`render-img-button-container ${hover ? "render-buttons-show" : ""}`}>
                    <img className="render-img-button" src={xHover ? xIconHover : xIcon} style={{filter: getFilter()}}
                    onMouseEnter={() => setXHover(true)} onMouseLeave={() => setXHover(false)} onClick={remove}/>
                </div> : null}
                <img className="render-img" src={renderImage} draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`}}/>
            </div> : null}
        </div>
    )
}

export default Render