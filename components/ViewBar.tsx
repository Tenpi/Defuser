import React, {useContext, useEffect, useState, useRef, useReducer} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
ViewImagesContext, UpdateImagesContext, GeneratorContext, ImageBrightnessContext, ImageContrastContext} from "../Context"
import functions from "../structures/Functions"
import downloadIcon from "../assets/icons/download.png"
import "./styles/viewbar.less"
import axios from "axios"
import JSZip from "jszip"
import path from "path"

interface ViewImageProps {
    img: string
    update: (img: string) => void
}

const ViewImage: React.FunctionComponent<ViewImageProps> = (props) => {
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const [hover, setHover] = useState(false)

    const download = async () => {
        const images = await axios.post("/similar-images", {image: props.img}).then((r) => r.data)
        const zip = new JSZip()
        for (let i = 0; i < images.length; i++) {
            const data = await fetch(images[i]).then((r) => r.arrayBuffer())
            zip.file(`${path.basename(images[i], path.extname(images[i]))}${path.extname(images[i])}`, data, {binary: true})
        }
        const filename = "images.zip"
        const blob = await zip.generateAsync({type: "blob"})
        const url = window.URL.createObjectURL(blob)
        functions.download(filename, url)
        window.URL.revokeObjectURL(url)
    }

    return (
        <div className="view-image-container" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
            <div className={`view-image-button-container ${hover ? "show-view-image-buttons" : ""}`}>
                <img className="view-image-button" src={downloadIcon} onClick={download} draggable={false}/>
            </div>
            <img className="view-image" src={props.img} onClick={() => props.update(props.img)} draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`}}/>
        </div>
    )
}

const ViewBar: React.FunctionComponent = (props) => {
    const [ignored, forceUpdate] = useReducer(x => x + 1, 0)
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {viewImages, setViewImages} = useContext(ViewImagesContext)
    const {updateImages, setUpdateImages} = useContext(UpdateImagesContext)
    const {generator, setGenerator} = useContext(GeneratorContext)
    const [viewImage, setViewImage] = useState("")
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getSaveKey = () => {
        if (generator === "novel ai") return "saved-novel-ai"
        if (generator === "holara ai") return "saved-holara-ai"
        return "saved"
    }

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedViewImages = localStorage.getItem("viewImages")
        if (savedViewImages) setViewImages(JSON.parse(savedViewImages))
    }, [])

    useEffect(() => {
        if (viewImages.length) localStorage.setItem("viewImages", JSON.stringify(viewImages))
    }, [viewImages])

    const updateViewImages = async (image: string) => {
        if (!image) return
        const viewImages = await axios.post("/similar-images", {image}).then((r) => r.data)
        setViewImages(viewImages)
        setViewImage(image)
    }

    useEffect(() => {
        if (updateImages) updateViewImages(viewImage)
    }, [updateImages])

    useEffect(() => {
        let saved = localStorage.getItem(getSaveKey()) || "[]" as any
        saved = JSON.parse(saved)
        if (saved.length) updateViewImages(saved[0])
    }, [generator])

    const generateJSX = () => {
        let jsx = [] as any
        let saved = localStorage.getItem(getSaveKey()) || "[]" as any
        saved = JSON.parse(saved)
        for (let i = 0; i < saved.length; i++) {
            jsx.push(<ViewImage img={saved[i]} update={updateViewImages}/>)
        }
        return jsx
    }

    return (
        <div className="viewbar" onMouseEnter={() => setEnableDrag(false)}>
            {generateJSX()}
        </div>
    )
}

export default ViewBar