import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext,
PreviewImageContext, TabContext, ImageBrightnessContext, ImageContrastContext} from "../Context"
import functions from "../structures/Functions"
import textIcon from "../assets/icons/text.png"
import textIconHover from "../assets/icons/text-hover.png"
import sourceIcon from "../assets/icons/source.png"
import sourceIconHover from "../assets/icons/source-hover.png"
import "./styles/trainimage.less"
import axios from "axios"

interface TrainImageProps {
    img: string
}

let timer = null as any
let clicking = false

const TrainImage: React.FunctionComponent<TrainImageProps> = (props) => {
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {previewImage, setPreviewImage} = useContext(PreviewImageContext)
    const {tab, setTab} = useContext(TabContext)
    const [hover, setHover] = useState(false)
    const [textHover, setTextHover] = useState(false)
    const [sourceHover, setSourceHover] = useState(false)

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    const showInFolder = () => {
        axios.post("/show-in-folder", {absolute: props.img.replace("/retrieve?path=", "")})
    }

    const preview = () => {
        setPreviewImage(props.img)
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

    const showText = async (event: any) => {
        event?.stopPropagation()
        await axios.post("/show-text", {image: props.img.replace("/retrieve?path=", "")})
    }

    const showSource = async (event: any) => {
        event?.stopPropagation()
        await axios.post("/show-source", {image: props.img.replace("/retrieve?path=", "")})
    }

    return (
        <div className="train-image-img-container" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} onClick={handleClick}>
            <div className={`train-image-img-button-container ${hover ? "train-image-buttons-show" : ""}`}>
                <img className="train-image-img-button" onMouseEnter={() => setSourceHover(true)} onClick={showSource} draggable={false}
                onMouseLeave={() => setSourceHover(false)} src={sourceHover ? sourceIconHover : sourceIcon} style={{filter: getFilter(), cursor: hover ? "pointer" : "default"}}/>
                <img className="train-image-img-button" onMouseEnter={() => setTextHover(true)} onClick={showText} draggable={false}
                onMouseLeave={() => setTextHover(false)} src={textHover ? textIconHover : textIcon} style={{filter: getFilter(), cursor: hover ? "pointer" : "default"}}/>
            </div>
            <img className="train-image-img" src={props.img} draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`}}/>
        </div>
    )
}

export default TrainImage