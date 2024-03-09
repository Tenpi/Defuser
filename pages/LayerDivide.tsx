import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, 
SiteLightnessContext, SocketContext, ImageBrightnessContext, ImageContrastContext,
TrainStartedContext, TrainCompletedContext, LayerDivideInputContext, ImageHueContext,
ImageSaturationContext} from "../Context"
import functions from "../structures/Functions"
import imgPlaceHolder from "../assets/images/img-placeholder.png"
import xIcon from "../assets/icons/x-alt.png"
import "./styles/traintag.less"
import fileType from "magic-bytes.js"
import Slider from "react-slider"
import axios from "axios"
import path from "path"

const LayerDivide: React.FunctionComponent = (props) => {
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
    const {layerDivideInput, setLayerDivideInput} = useContext(LayerDivideInputContext)
    const [hover, setHover] = useState(false)
    const [img, setImg] = useState(null) as any
    const [loops, setLoops] = useState(3)
    const [clusters, setClusters] = useState(10)
    const [clusterThreshold, setClusterThreshold] = useState(15)
    const [blurSize, setBlurSize] = useState(5)
    const [area, setArea] = useState(20000)
    const [divideMode, setDivideMode] = useState("color_base_mode")
    const [layerMode, setLayerMode] = useState("composite")
    const [downscale, setDownscale] = useState("")
    const progressBarRef = useRef(null) as React.RefObject<HTMLDivElement>
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedImage = localStorage.getItem("layerDivideInput")
        if (savedImage) setLayerDivideInput(savedImage)
        const savedLoops = localStorage.getItem("loops")
        if (savedLoops) setLoops(Number(savedLoops))
        const savedClusters = localStorage.getItem("clusters")
        if (savedClusters) setClusters(Number(savedClusters))
        const savedClusterThreshold = localStorage.getItem("clusterThreshold")
        if (savedClusterThreshold) setClusterThreshold(Number(savedClusterThreshold))
        const savedBlurSize = localStorage.getItem("blurSize")
        if (savedBlurSize) setBlurSize(Number(savedBlurSize))
        const savedDivideMode = localStorage.getItem("divideMode")
        if (savedDivideMode) setDivideMode(savedDivideMode)
        const savedLayerMode = localStorage.getItem("layerMode")
        if (savedLayerMode) setLayerMode(savedLayerMode)
        const savedArea = localStorage.getItem("area")
        if (savedArea) setArea(Number(savedArea))
        const savedDownscale = localStorage.getItem("downscale")
        if (savedDownscale) setDownscale(savedDownscale)
    }, [])

    useEffect(() => {
        try {
            localStorage.setItem("loops", String(loops))
            localStorage.setItem("clusters", String(clusters))
            localStorage.setItem("clusterThreshold", String(clusterThreshold))
            localStorage.setItem("blurSize", String(blurSize))
            localStorage.setItem("divideMode", String(divideMode))
            localStorage.setItem("layerMode", String(layerMode))
            localStorage.setItem("area", String(area))
            localStorage.setItem("downscale", String(downscale))
            localStorage.setItem("layerDivideInput", String(layerDivideInput))
        } catch {
            // ignore
        }
    }, [layerDivideInput, loops, clusters, clusterThreshold, blurSize, divideMode, layerMode, area, downscale])

    useEffect(() => {
        if (!socket) return
        const startTrain = () => {
            setTrainStarted(true)
            setTrainCompleted(false)
        }
        const completeTrain = async () => {
            setTrainCompleted(true)
            setTrainStarted(false)
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
                        setLayerDivideInput(link)
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
        setLayerDivideInput("")
        setImg(null)
    }

    const loadImages = async () => {
        if (!layerDivideInput) return
        const image = document.createElement("img")
        await new Promise<void>((resolve) => {
            image.onload = () => resolve()
            image.src = layerDivideInput
        })
        setImg(image)
    }
    useEffect(() => {
        loadImages()
    }, [layerDivideInput])

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

    const divide = async () => {
        const json = {} as any
        json.image = functions.cleanBase64(layerDivideInput)
        json.loops = loops
        json.clusters = clusters
        json.cluster_threshold = clusterThreshold
        json.blur_size = blurSize
        json.divide_mode = divideMode
        json.layer_mode = layerMode
        json.area = area
        json.downscale = downscale
        await axios.post("/layer-divide", json)
    }

    const interrupt = async () => {
        axios.post("/interrupt-misc")
    }

    const openFolder = async () => {
        await axios.post("/open-folder", {path: `outputs/local/psd`})
    }

    const reset = () => {
        if (divideMode === "color_base_mode") {
            setLoops(3)
            setClusters(10)
            setClusterThreshold(15)
            setBlurSize(5)
        } else {
            setArea(20000)
        }
        setDownscale("")
    }

    return (
        <div className="train-tag" onMouseEnter={() => setEnableDrag(false)} style={{flexDirection: "row", alignItems: "center"}}>
            <div className="train-tag-column">
                <div className="options-bar-img-input">
                    <span className="options-bar-text">Img Input</span>
                    <label htmlFor="img" className="options-bar-img-container" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
                        <div className={`options-bar-img-button-container ${layerDivideInput && hover ? "show-options-bar-img-buttons" : ""}`}>
                            <img className="options-bar-img-button" src={xIcon} onClick={removeImage} style={{filter: getFilter()}} draggable={false}/>
                        </div>
                        {layerDivideInput ? 
                        <canvas ref={ref} className="options-bar-img" draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%) hue-rotate(${imageHue - 180}deg) saturate(${imageSaturation}%)`}}></canvas> :
                        <img className="options-bar-img" src={imgPlaceHolder} style={{filter: getFilter()}} draggable={false}/>}
                    </label>
                    <input id="img" type="file" onChange={(event) => loadImage(event)}/>
                </div>
                <div className="shade-sketch-box" style={{border: "none"}}>
                    <div className="shade-sketch-box-row" style={{background: "transparent"}}>
                        <span className="train-tag-settings-title">Downscale:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={downscale} onChange={(event) => setDownscale(event.target.value)} style={{width: "70px"}}/>
                    </div>
                </div>
                <div className="shade-sketch-box">
                    <div className="shade-sketch-box-row">
                        <button className="shade-sketch-button" style={{backgroundColor: divideMode === "color_base_mode" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setDivideMode("color_base_mode")}>Color Base</button>
                        <button className="shade-sketch-button" style={{backgroundColor: divideMode === "segment_mode" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setDivideMode("segment_mode")}>Segment</button>
                    </div>
                </div>
                <div className="shade-sketch-box">
                    <div className="shade-sketch-box-row">
                        <button className="shade-sketch-button" style={{backgroundColor: layerMode === "normal" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setLayerMode("normal")}>Normal</button>
                        <button className="shade-sketch-button" style={{backgroundColor: layerMode === "composite" ? "var(--buttonBGStop)" : "var(--buttonBG)"}} onClick={() => setLayerMode("composite")}>Composite</button>
                    </div>
                </div>
                <button className="train-tag-button" onClick={() => trainStarted ? interrupt() : divide()} style={{backgroundColor: trainStarted ? "var(--buttonBGStop)" : "var(--buttonBG)", marginLeft: "0px"}}>{trainStarted ? "Stop" : "Divide"}</button>
            </div>
            <div className="train-tag-column" style={{marginTop: "-40px"}}>
                {divideMode === "color_base_mode" ? <>
                <div className="train-slider-row">
                    <span className="train-slider-text">Loops</span>
                    <Slider className="train-slider" trackClassName="train-slider-track" thumbClassName="train-slider-thumb" onChange={(value) => setLoops(value)} min={1} max={20} step={1} value={loops}/>
                    <span className="train-slider-text-value">{loops}</span>
                </div>
                <div className="train-slider-row">
                    <span className="train-slider-text">Clusters</span>
                    <Slider className="train-slider" trackClassName="train-slider-track" thumbClassName="train-slider-thumb" onChange={(value) => setClusters(value)} min={1} max={50} step={1} value={clusters}/>
                    <span className="train-slider-text-value">{clusters}</span>
                </div>
                <div className="train-slider-row">
                    <span className="train-slider-text">Cluster Threshold</span>
                    <Slider className="train-slider" trackClassName="train-slider-track" thumbClassName="train-slider-thumb" onChange={(value) => setClusterThreshold(value)} min={1} max={50} step={1} value={clusterThreshold}/>
                    <span className="train-slider-text-value">{clusterThreshold}</span>
                </div>
                <div className="train-slider-row">
                    <span className="train-slider-text">Blur Size</span>
                    <Slider className="train-slider" trackClassName="train-slider-track" thumbClassName="train-slider-thumb" onChange={(value) => setBlurSize(value)} min={1} max={20} step={1} value={blurSize}/>
                    <span className="train-slider-text-value">{blurSize}</span>
                </div></> : null}
                {divideMode === "segment_mode" ?
                <div className="train-slider-row">
                    <span className="train-slider-text">Area</span>
                    <Slider className="train-slider" trackClassName="train-slider-track" thumbClassName="train-slider-thumb" onChange={(value) => setArea(value)} min={100} max={100000} step={100} value={area}/>
                    <span className="train-slider-text-value">{area}</span>
                </div> : null}
                <div className="train-slider-row">
                    <button className="train-tag-button" onClick={() => openFolder()} style={{marginLeft: "0px"}}>Open</button>
                    <button className="train-tag-button" onClick={() => reset()}>Reset</button>
                </div>
            </div>
        </div>
    )
}

export default LayerDivide