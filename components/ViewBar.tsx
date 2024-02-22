import React, {useContext, useEffect, useState, useRef, useReducer} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, ViewImagesContext, UpdateImagesContext} from "../Context"
import functions from "../structures/Functions"
import "./styles/viewbar.less"
import axios from "axios"

const ViewBar: React.FunctionComponent = (props) => {
    const [ignored, forceUpdate] = useReducer(x => x + 1, 0)
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {viewImages, setViewImages} = useContext(ViewImagesContext)
    const {updateImages, setUpdateImages} = useContext(UpdateImagesContext)
    const [viewImage, setViewImage] = useState("")
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

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
        let saved = localStorage.getItem("saved") || "[]" as any
        saved = JSON.parse(saved)
        if (saved.length) updateViewImages(saved[0])
    }, [])

    const generateJSX = () => {
        let jsx = [] as any
        let saved = localStorage.getItem("saved") || "[]" as any
        saved = JSON.parse(saved)
        for (let i = 0; i < saved.length; i++) {
            jsx.push(<img onClick={() => updateViewImages(saved[i])} className="view-image" src={saved[i]}/>)
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